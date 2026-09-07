"""Inference: read a folder of face images, write a CSV of attack scores.

    python infer.py --input /path/to/images [--output outputs.csv] [--artefacts artefacts] [--no-recursive]

Any file that Pillow can open is scored, whatever its extension: PNG, JPEG, HEIC/HEIF (what iPhones save by
default), BMP, TIFF, WebP, GIF, PPM. Grayscale, palette and RGBA images are converted to RGB. EXIF rotation is
applied. Files that are not images are listed and skipped; images that fail are kept in the CSV with an error.

Output CSV columns
  file        path relative to the input folder
  score       0..1, higher = more likely a screen attack (blank if the image could not be scored)

The console summary reports the rest: how many files were found, skipped, or failed, the median time per image,
which images had no detectable face (scored on the image centre, less reliable), and any errors.

Thresholds at 1 %, 5 % and 10 % BPCER are in artefacts/config.json; they are printed but not applied, because the
operating point is the deployer's decision.

Pipeline (identical to training, see pad/data.py): open, apply EXIF rotation, RGB, drop metadata, detect the face
(OpenCV Haar cascade, image-centre fallback), crop a square around it at native resolution, cut 36 patches of
128 px on a fixed grid, score each with the network, average.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pad import data as P
from pad.model import load_model, to_tensor

import pillow_heif                     # iPhone HEIC/HEIF: registered so Image.open handles them like any other format
pillow_heif.register_heif_opener()

SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def open_image(path):
    """RGB PIL image with EXIF rotation applied and nothing but pixels kept. Raises UnidentifiedImageError for non-images."""
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("P", "PA", "LA", "RGBA", "I;16", "I", "F", "CMYK", "YCbCr", "L", "1"):
            im = im.convert("RGB") if im.mode != "RGBA" else Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im).convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        return Image.fromarray(np.asarray(im))          # drops EXIF/ICC/XMP; only the pixel array survives


@torch.no_grad()
def score_image(path, model, cfg):
    t0 = time.perf_counter()
    im = open_image(path)
    box = P.face_box(im)
    reg = P.face_region(im, box)
    grid = P.patch_grid(reg, cfg["patch"])
    x = torch.stack([to_tensor(P.crop_patch(reg, xy, cfg["patch"])) for xy in grid])
    pr = torch.sigmoid(model(x).squeeze(1)).numpy()
    return dict(score=float(pr.mean()), face_found=bool(box[3]), n_patches=int(len(grid)), seconds=round(time.perf_counter() - t0, 3), error="")


def main():
    ap = argparse.ArgumentParser(description="PAD screen-attack scorer: folder of images in, CSV of scores out")
    ap.add_argument("--input", required=True, help="folder of images (or a single image file)")
    ap.add_argument("--output", default="outputs.csv", help="CSV to write (default outputs.csv in the current folder)")
    ap.add_argument("--artefacts", default=str(Path(__file__).resolve().parent / "artefacts"))
    ap.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    art = Path(a.artefacts)
    cfg = json.load(open(art / "config.json"))
    model = load_model(art / "model.pt")

    root = Path(a.input)
    if root.is_file():
        files, base = [root], root.parent
    else:
        it = root.rglob("*") if not a.no_recursive else root.glob("*")
        files, base = sorted(p for p in it if p.is_file() and p.name not in SKIP_NAMES and not p.name.startswith(".")), root
    if not files:
        sys.exit(f"no files found under {a.input}")
    if root.is_file():
        print(f"1 file: {root}")
    else:
        print(f"{len(files)} file{'s' if len(files) != 1 else ''} found under {root}" + ("" if a.no_recursive else ", including subfolders"))
    print("scoring every file Pillow can open; score is 0..1, higher = more likely a screen attack")

    rows, skipped = [], []
    for p in tqdm(files, unit="img", ncols=80):
        rel = str(p.relative_to(base))
        try:
            rows.append(dict(file=rel, **score_image(p, model, cfg)))
        except UnidentifiedImageError:
            skipped.append(rel)                                   # not an image: leave it out of the CSV
        except Exception as e:                                    # an image that failed: keep the row, record why
            rows.append(dict(file=rel, score=float("nan"), face_found=False, n_patches=0, seconds=0.0, error=f"{type(e).__name__}: {e}"))
    full = pd.DataFrame(rows, columns=["file", "score", "face_found", "n_patches", "seconds", "error"])
    df = full[["file", "score"]]
    df.to_csv(a.output, index=False)

    print(f"\n{len(df)} image{'s' if len(df) != 1 else ''} scored, {int(full.score.isna().sum())} failed, {len(skipped)} non-image file{'s' if len(skipped) != 1 else ''} skipped"
          + (f" ({', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})" if skipped else ""))
    if full.seconds.gt(0).any():
        print(f"median {full.seconds[full.seconds > 0].median():.2f} s per image")
    nf = full[~full.face_found & full.score.notna()].file.tolist()
    if nf:
        print(f"{len(nf)} scored without a detected face (image centre used, less reliable): {', '.join(nf[:5])}{'...' if len(nf) > 5 else ''}")
    bad = full[full.score.isna()]
    for r in bad.itertuples():
        print(f"failed: {r.file}: {r.error}")
    print(f"wrote {Path(a.output).resolve()}\n")
    print("first rows:")
    print(df.head(10).to_string(index=False))
    thr = cfg.get("thresholds", {})
    if thr:
        print("reference thresholds from development data (not applied):", {k: round(v, 3) for k, v in thr.items()})


if __name__ == "__main__":
    main()
