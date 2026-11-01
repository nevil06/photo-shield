"""Face identity: detection, alignment, ArcFace embedding, cosine.

Why this file exists twice over:
  * **evaluation** needs the *exact* published ArcFace (ONNX, buffalo_l /
    w600k_r50). onnxruntime gives bit-stable numbers and no gradients.
  * **the attack** needs a *differentiable* embedder, because the identity loss
    is computed on an edited image and the gradient must reach our pixels.

The honest way to have both without lying about the model: convert the *same*
ONNX file to torch (`onnx2torch`) and prove the two agree (test_arcface.py,
gated). If they ever disagree, the attack is optimising a different model than
the one we report — that would be a silent fraud, so it is a test.

Everything heavy is imported lazily so `import photoshield` stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..config import WEIGHTS_DIR

if TYPE_CHECKING:
    import torch

# ArcFace's canonical 5-point template for a 112x112 crop (left eye, right eye,
# nose tip, left mouth corner, right mouth corner). Every ArcFace variant uses
# these same coordinates - that is what makes embeddings comparable.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

DEFAULT_REC_MODEL = WEIGHTS_DIR / "buffalo_l" / "w600k_r50.onnx"
DEFAULT_DET_MODEL = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"


# ---------------------------------------------------------------- alignment
def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (scale + rotation + translation).

    Umeyama (1991): find s, R, t minimising ||dst - (s*R@src + t)||. Returns the
    2x3 affine matrix. With 5 well-spread landmarks and no outliers this is
    exact, which is why we do not need RANSAC.

    Closed form: cov = D^T S / n; SVD -> U, sigma, Vt; R = U @ diag(1, d) @ Vt
    with d = sign(det(U Vt)) to forbid reflections; scale = tr(diag(sigma) D) /
    var(src).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"need matching (N,2) arrays, got {src.shape} and {dst.shape}")
    if src.shape[0] < 2:
        raise ValueError("need at least 2 point pairs")
    n = src.shape[0]

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / n  # 2x2
    U, sigma, Vt = np.linalg.svd(cov)

    d = float(np.sign(np.linalg.det(U @ Vt))) or 1.0  # no reflection
    R = U @ np.diag([1.0, d]) @ Vt

    var_s = float((S**2).sum() / n)
    scale = 1.0 if var_s < 1e-12 else float((sigma * np.array([1.0, d])).sum() / var_s)
    t = mu_d - scale * (R @ mu_s)

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 matrix to (N,2) points. Used by tests and by nobody else."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:, :2].T + M[:, 2]


def affine_theta(M: np.ndarray, in_size: tuple[int, int], out_size: int = 112) -> np.ndarray:
    """OpenCV's src->dst 2x3 matrix -> theta for `torch.nn.functional.affine_grid`.

    Why this exists: the identity loss must see the *same* landmark-aligned
    112x112 view as ArcFace evaluation. `cv2.warpAffine` is not differentiable,
    so we convert the same affine into a sampling grid that `grid_sample` can
    differentiate through.

    Coordinate chain (align_corners=False):
        output pixel -> normalized output -> input pixel -> normalized input
    `M` maps input pixels to output pixels; grid_sample needs the inverse.
    """
    w, h = (float(v) for v in in_size)
    A = np.eye(3, dtype=np.float64)
    A[:2] = np.asarray(M, dtype=np.float64)
    Ainv = np.linalg.inv(A)  # output pixel -> input pixel

    # pixel -> normalized, for an input of size (w, h)
    N_in = np.array(
        [[2.0 / w, 0.0, 1.0 / w - 1.0], [0.0, 2.0 / h, 1.0 / h - 1.0], [0.0, 0.0, 1.0]]
    )
    # normalized -> pixel, for an output of size (out, out)
    P_out = np.array(
        [
            [out_size / 2.0, 0.0, (out_size - 1) / 2.0],
            [0.0, out_size / 2.0, (out_size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    theta = N_in @ Ainv @ P_out
    return theta[:2].astype(np.float32)


def align_face(img: Image.Image, kps: np.ndarray, size: int = 112) -> Image.Image:
    """Warp a photo so the 5 landmarks land on the ArcFace template.

    If `kps` are already the template, this is (almost) a no-op - which is the
    test that catches a transposed matrix or an RGB/BGR mix-up.
    """
    import cv2  # local: cv2 is only needed when a face is actually aligned

    M = umeyama_similarity(kps, ARCFACE_TEMPLATE[: len(kps)])
    arr = np.asarray(img.convert("RGB"))
    out = cv2.warpAffine(
        arr, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(out)


# ---------------------------------------------------------------- embedding
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalised vectors; plain dot product otherwise."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("zero-length embedding")
    return float(a @ b / (na * nb))


class ArcFaceONNX:
    """The evaluation embedder: exact published weights, no gradients.

    Preprocessing matters and is the usual source of "my cosine is 0.2" bugs:
    the model was trained on **BGR**, 112x112, scaled to [-1, 1] via
    (x - 127.5) / 127.5. Our pipeline is RGB, hence the channel flip.
    """

    def __init__(self, model_path: str | Path = DEFAULT_REC_MODEL, device: str = "cpu") -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ArcFace ONNX not found: {path}\n"
                "Download the InsightFace buffalo_l pack into weights/buffalo_l/ first "
                "(see README / docs_private/PROGRESS.md)."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else [
            "CPUExecutionProvider"
        ]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed_aligned(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """112x112 RGB aligned face -> L2-normalised embedding (float32, 512-d)."""
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 64:
            raise ValueError(f"expected an aligned face image, got {arr.shape}")
        bgr = arr[:, :, ::-1]  # RGB -> BGR (model contract)
        blob = ((bgr - 127.5) / 127.5).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        return (out / np.linalg.norm(out)).astype(np.float32)


class TorchArcFace:
    """The attack embedder: the *same* ONNX weights, converted to torch.

    Why not train our own: identity must match the metric we report. Why not
    skip identity loss: an encoder attack only degrades the edit; it does not
    specifically change who the edited face looks like.
    """

    def __init__(self, model_path: str | Path = DEFAULT_REC_MODEL, device: str = "cpu") -> None:
        from onnx2torch import convert

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ArcFace ONNX not found: {path}\n"
                "Convert it with onnx2torch after downloading buffalo_l."
            )
        import onnx

        self.device = device
        # convert(str) runs shape inference via a NamedTemporaryFile that stays
        # open, and Windows then refuses to reopen it (PermissionError). Loading
        # the ModelProto ourselves skips that path entirely.
        try:
            net = convert(onnx.load(str(path)))
        except Exception as exc:  # pragma: no cover - depends on the ONNX ops
            raise RuntimeError(
                f"onnx2torch could not convert {path.name}: {exc}. The identity "
                "loss cannot use these weights, so do not report identity results."
            ) from exc
        self.net = net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)  # frozen: we differentiate w.r.t. the image

    def forward(self, x01: "torch.Tensor") -> "torch.Tensor":
        """(B,3,112,112) RGB in [0,1] -> (B,D) L2-normalised, differentiable."""
        import torch.nn.functional as F

        x = x01[:, [2, 1, 0]]  # RGB -> BGR
        x = (x * 255.0 - 127.5) / 127.5
        return F.normalize(self.net(x), dim=-1)

    __call__ = forward


# ---------------------------------------------------------------- detection
class YuNetDetector:
    """Tiny face detector with 5 landmarks (OpenCV YuNet, ~230 KB).

    Why YuNet and not SCRFD/insightface: `pip install insightface` needs a C++
    toolchain on Windows; YuNet ships with opencv (no build) and already returns
    the 5 points ArcFace alignment needs.
    """

    def __init__(self, model_path: str | Path = DEFAULT_DET_MODEL, min_score: float = 0.6) -> None:
        import cv2

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YuNet ONNX not found: {path}\n"
                "Download face_detection_yunet_2023mar.onnx into weights/ first."
            )
        self.min_score = min_score
        self._det = cv2.FaceDetectorYN_create(str(path), "", (320, 320), min_score, 0.3, 5000)

    def detect(self, img: Image.Image) -> list[dict]:
        """Return faces as {'box': (x,y,w,h), 'kps': (5,2), 'score': float}."""
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # cv2 wants BGR
        h, w = arr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(arr)
        if faces is None:
            return []
        out = []
        for f in faces:
            if float(f[-1]) < self.min_score:
                continue
            out.append(
                {
                    "box": tuple(float(v) for v in f[:4]),
                    "kps": f[4:14].reshape(5, 2).astype(np.float32),
                    "score": float(f[-1]),
                }
            )
        return out

    def largest(self, img: Image.Image) -> dict | None:
        faces = self.detect(img)
        if not faces:
            return None
        return max(faces, key=lambda f: f["box"][2] * f["box"][3])


# ---------------------------------------------------------------- facade
class IdentityModel:
    """Detect the largest face, align it, embed it. One object for eval + attack."""

    def __init__(
        self,
        device: str = "cpu",
        rec_path: str | Path = DEFAULT_REC_MODEL,
        det_path: str | Path = DEFAULT_DET_MODEL,
        torch_embedder: bool = False,
    ) -> None:
        self.detector = YuNetDetector(det_path)
        self.embedder = (
            TorchArcFace(rec_path, device=device) if torch_embedder else ArcFaceONNX(rec_path, device=device)
        )
        self.torch_embedder = torch_embedder

    def embed_image(self, img: Image.Image) -> tuple[np.ndarray | "torch.Tensor", dict]:
        """Full photo -> embedding + metadata. Raises if no face is found."""
        face = self.detector.largest(img)
        if face is None:
            raise ValueError("no face detected; crop tighter or lower the detector threshold")
        aligned = align_face(img, face["kps"])
        if self.torch_embedder:
            import torch

            from ..data import to_tensor

            with torch.no_grad():
                emb = self.embedder(to_tensor(aligned).to(self.embedder.device))[0]
            return emb, {"aligned": aligned, **face}
        return self.embedder.embed_aligned(aligned), {"aligned": aligned, **face}

    def crop_box(self, img: Image.Image, pad: float = 0.45) -> tuple[int, int, int, int] | None:
        """Square box around the largest face, padded, in pixel coordinates.

        Kept for diagnostics and the old tests. The *loss* now uses
        `alignment_theta`, because a padded square box is not what ArcFace sees.
        """
        face = self.detector.largest(img)
        if face is None:
            return None
        x, y, w, h = face["box"]
        cx, cy = x + w / 2, y + h / 2
        half = max(w, h) * (0.5 + pad)
        w_img, h_img = img.size
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(w_img, cx + half)), int(min(h_img, cy + half))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return x0, y0, x1, y1

    def alignment_theta(self, img: Image.Image, out_size: int = 112) -> np.ndarray | None:
        """Landmark-aligned sampling matrix for a differentiable 112x112 crop.

        Same Umeyama affine as `align_face`/insightface.norm_crop, but expressed
        as an `affine_grid` theta so gradients can flow back to the pixels.
        Returns None when no face is detected; callers must fail loudly.
        """
        face = self.detector.largest(img)
        if face is None:
            return None
        M = umeyama_similarity(face["kps"], ARCFACE_TEMPLATE[: len(face["kps"])])
        return affine_theta(M, img.size, out_size)
