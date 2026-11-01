"""Phase 5 tests for the parts of evaluate.py that need no editor weights."""

import pytest
import torch

from photoshield.data import synthetic_image, to_tensor
from photoshield.evaluate import (
    POSTPROCESS,
    markdown_table,
    measured_eps,
    postprocess,
    sign_noise,
    summarize,
)


@pytest.mark.parametrize("kind", POSTPROCESS)
def test_postprocess_keeps_size(kind):
    img = synthetic_image(96)
    assert postprocess(img, kind).size == img.size


def test_postprocess_none_is_identity():
    img = synthetic_image(64)
    assert torch.equal(to_tensor(postprocess(img, "none")), to_tensor(img))


def test_postprocess_changes_pixels():
    img = synthetic_image(64)
    for kind in ("jpeg75", "resize50", "crop90"):
        assert not torch.equal(to_tensor(postprocess(img, kind)), to_tensor(img)), kind


def test_postprocess_unknown_raises():
    with pytest.raises(ValueError):
        postprocess(synthetic_image(64), "webp")


def test_sign_noise_budget_and_determinism():
    img = synthetic_image(64)
    eps = 4 / 255
    a, b = sign_noise(img, eps, seed=1), sign_noise(img, eps, seed=1)
    assert torch.equal(to_tensor(a), to_tensor(b))
    # uint8 rounding can add at most half a grey level
    assert measured_eps(img, a) <= eps + 0.5 / 255 + 1e-6


def test_summarize_drops():
    rows = [
        {"postprocess": "none", "input": "original", "cos_edit_to_source": 0.8},
        {"postprocess": "none", "input": "protected", "cos_edit_to_source": 0.3},
        {"postprocess": "none", "input": "noise", "cos_edit_to_source": 0.75},
        {"postprocess": "jpeg75", "input": "original", "cos_edit_to_source": 0.4},
        {"postprocess": "jpeg75", "input": "protected", "cos_edit_to_source": None},
    ]
    s = {r["postprocess"]: r for r in summarize(rows)}
    assert s["none"]["drop_protected"] == pytest.approx(0.5)
    assert s["none"]["drop_noise"] == pytest.approx(0.05)
    assert s["none"]["baseline_valid"] is True
    assert s["jpeg75"]["drop_protected"] is None  # no face found -> no number, not 0
    assert s["jpeg75"]["baseline_valid"] is False


def test_markdown_table_shape():
    md = markdown_table([{"a": 1.23456, "b": None, "c": True}], ["a", "b", "c"])
    lines = md.splitlines()
    assert lines[0] == "| a | b | c |"
    assert lines[2] == "| 1.2346 | — | yes |"
