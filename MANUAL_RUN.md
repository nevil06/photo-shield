# PhotoShield: run it yourself

These are the steps to protect a photo and test it against a real AI editor, on
this laptop (Windows, CPU only). Run every command in **PowerShell** from the
project folder. The times below were measured on this laptop.

## 0. Before you start

- Close Chrome, Roblox and other heavy apps. A 512 px protection run needs
  about **6 GB of free RAM**.
- Keep the laptop plugged in, and don't let it sleep.
- To stop any command, press **Ctrl+C** in its window.

Open PowerShell and run:

```powershell
cd "C:\Users\nevil\Desktop\photo pro"
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = "1"        # models are already downloaded, so don't contact the internet
$env:PYTHONWARNINGS = "ignore"   # hide harmless library warnings
$env:PYTHONUNBUFFERED = "1"      # write logs as the run goes, not only at the end
```

Run these again every time you open a new PowerShell window.

## 1. Check that the code works (about 10 s)

```powershell
python -m pytest -q
```

Expected: `68 passed, 3 skipped`. If anything fails, stop here.

## 2. Protect the test photo (about 40 min per run)

The test photo is `assets\samples\astronaut_face512.png`, a 512 px crop around
the face. The two runs use the two attack modes:

| `--target` | What it does |
|---|---|
| `none` | Original method: pushes the AI's view of the photo away from the original |
| `pattern` | New method: pulls the AI's view of the photo toward a striped decoy pattern |

Run 1:

```powershell
python -m photoshield.cli protect assets\samples\astronaut_face512.png `
    -o results\phase5\face512_eps4_w2_none.png `
    --eps-255 4 --size 512 --steps 50 --w-identity 2 --target none `
    2>&1 | Tee-Object results\phase5\protect_face512_none.log
```

Run 2:

```powershell
python -m photoshield.cli protect assets\samples\astronaut_face512.png `
    -o results\phase5\face512_eps4_w2_pattern.png `
    --eps-255 4 --size 512 --steps 50 --w-identity 2 --target pattern `
    2>&1 | Tee-Object results\phase5\protect_face512_pattern.log
```

You'll see progress about every 10 steps. At the end, check:

- `L-inf` must be `4.00/255` or lower. This means no pixel changed by more
  than 4 levels.
- `PSNR` should be above 35 dB and `LPIPS` below about 0.01. This means the
  change is invisible.
- `identity cosine` should end far below where it started. This is the stand-in
  test only, not the real editor.

For a quick trial first, use `--size 256 --steps 20`. That takes about 5 min,
but the result is weaker.

## 3. Test against a real AI editor (about 35 min per run)

The editor is InstructPix2Pix with the prompt "turn her into an oil painting".
On the unprotected photo it keeps the face at a score of **0.93**, so it's a
valid test. Each run edits the original, the protected photo and a random-noise
copy. It does this three times: unchanged, after JPEG quality 75 (like social
media) and after a 50 % resize.

Test run 1:

```powershell
python -m photoshield.evaluate --editor ip2p `
    --original assets\samples\astronaut_face512.png `
    --protected results\phase5\face512_eps4_w2_none.png `
    --prompt "turn her into an oil painting" --steps 20 `
    --postprocess none jpeg75 resize50 `
    --out-dir results\phase5\ip2p_none `
    2>&1 | Tee-Object results\phase5\eval_ip2p_none.log
```

Test run 2:

```powershell
python -m photoshield.evaluate --editor ip2p `
    --original assets\samples\astronaut_face512.png `
    --protected results\phase5\face512_eps4_w2_pattern.png `
    --prompt "turn her into an oil painting" --steps 20 `
    --postprocess none jpeg75 resize50 `
    --out-dir results\phase5\ip2p_pattern `
    2>&1 | Tee-Object results\phase5\eval_ip2p_pattern.log
```

Results are saved after each post-process, so stopping halfway keeps what's done.

## 4. Read the result

Open `results\phase5\ip2p_none\results.md` and `results\phase5\ip2p_pattern\results.md`.
In the **Summary** table, each row is one post-process:

| Column | Meaning | Good result |
|---|---|---|
| `cos_edit_original` | Face-match score of the edited unprotected photo | about 0.9 |
| `cos_edit_protected` | Face-match score of the edited protected photo | as low as possible |
| `drop_protected` | How much protection lowered the score | **large, e.g. above 0.3** |
| `drop_noise` | How much random noise lowered the score | **small, e.g. below 0.1** |
| `baseline_valid` | Whether the test counts | must be `yes` |

**The protection works against this editor** only if `drop_protected` is large
**and** `drop_noise` is small. If both are small, it doesn't work against this
editor. If both are large, any noise would do the same, and the attack isn't
the cause.

The `jpeg75` and `resize50` rows show whether the protection survives social-media
compression. Also look at the images, for example `none_original.png` next to
`none_protected.png`.

## 5. Optional: test SD img2img as well

At strength 0.3, img2img already changes the face on the unprotected photo (score
0.42, below the 0.5 the test needs). Find a gentler setting first:

```powershell
python scripts\phase5_baseline.py --image assets\samples\astronaut_face512.png `
    --editors img2img --strengths 0.15 0.2 --out results\phase5\baseline_low
```

If one strength scores **0.5 or above**, run section 3 with
`--editor img2img --strength <that value> --prompt "portrait photo of a woman, detailed, studio lighting"`.

## 6. Protect your own photo

Put the photo in `assets\samples\`. That folder is never committed to git. Then run:

```powershell
python -m photoshield.cli protect assets\samples\me.jpg --eps-255 4 --size 512 --steps 50 --w-identity 2
```

The output is `assets\samples\me_protected.png`. Post the PNG. Converting it to
JPEG can weaken the protection.

`--w-identity 2` needs a clearly visible face. If you get "no face detected",
use `--w-identity 0`.

## 7. Web page

```powershell
python -m photoshield.app
```

Open http://127.0.0.1:7860. Don't use `--share` on this laptop: it creates a
public link with no password.

## What to send back

Send the two `results.md` files from section 4, plus the last ~15 lines of each
protection log from section 2.
