"""Input pipeline. Everything a model sees goes through here, at training and at inference.

    load_image(path)                  open, apply EXIF rotation, RGB. Metadata is dropped because only pixels leave.
    face_box(im)                      (cx, cy, side) of the face, OpenCV Haar cascade; centre-of-image fallback
    face_region(im, box)              square crop around the face at NATIVE resolution (no resize)
    face_view(im, box, size)          the whole face resized to a fixed size (global view)
    patch_grid(region, patch)         deterministic patch positions for evaluation
    patch_random(region, patch, n)    random patch positions for training
    augment(patch, rng)               capture-variation augmentation (colour, blur, noise, JPEG, rescale)
    FaceCache                         caches face boxes to derived/faces.csv and native face regions to cache/faces/

Why patches at native resolution: the screen pixel grid, banding and fine noise live at the camera's pixel scale
and are destroyed by resizing the whole face to 224 or 512 px (see the report, section 2). A 128 px patch keeps them.

Known confound: a 128 px patch covers about 15 % of a bona fide face but about 6 % of an attack face, because the
attacks were photographed at higher pixel counts. Random rescaling in augment() blurs that difference; the phone
transfer check is the test of whether it was enough.
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

import os

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PAD_DATA", ROOT / "data"))   # expects DATA/Bonafide and DATA/Screens

_CASCADES = None


def _cascades():
    global _CASCADES
    if _CASCADES is None:
        _CASCADES = [cv2.CascadeClassifier(cv2.data.haarcascades + f)
                     for f in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml")]
        assert not any(c.empty() for c in _CASCADES)
    return _CASCADES


# ---------------------------------------------------------------- loading and face localisation
def load_image(path):
    """RGB PIL image with EXIF rotation applied. Nothing but pixels survives this call."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return Image.fromarray(np.asarray(im))          # drops im.info (EXIF, ICC, XMP)


def face_box(im, min_frac=0.15):
    """(cx, cy, side, found). Most confident frontal face of adequate size; image centre if none."""
    w, h = im.size
    s = 640 / max(w, h)
    small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    g = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
    for C in _cascades():
        faces, _, weights = C.detectMultiScale3(g, scaleFactor=1.05, minNeighbors=3, minSize=(48, 48), outputRejectLevels=True)
        cand = [(wt, f) for f, wt in zip(faces, np.ravel(weights)) if max(f[2], f[3]) >= min_frac * min(g.shape)]
        if cand:
            x, y, fw, fh = max(cand, key=lambda c: c[0])[1]
            return (x + fw / 2) / s, (y + fh / 2) / s, max(fw, fh) / s, True
    return w / 2, h / 2, min(w, h) / 1.6, False


def _square(im, cx, cy, side):
    w, h = im.size
    side = int(min(side, w, h))
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return im.crop((x0, y0, x0 + side, y0 + side))


def face_region(im, box, margin=0.3):
    """Square around the face, side = face side * (1 + margin), native pixels. The patch source."""
    cx, cy, side, _ = box
    return _square(im, cx, cy, side * (1 + margin))


def face_view(im, box, size=256, margin=0.6):
    """Whole face at a fixed size. For the global branch and for anomaly-detection embeddings."""
    cx, cy, side, _ = box
    return _square(im, cx, cy, side * (1 + margin)).resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------- patches
def patch_grid(region, patch=128, per_side=6):
    """Exactly per_side**2 patch top-left corners for every image, evenly spaced over the region.

    Fixed count on purpose: attacks have larger face regions than bona fide in this dataset, so a count that
    depended on region size would leak class information. Patches overlap when the region is small.
    """
    W = region.size[0]
    if W < patch:                                          # tiny region: every position is (0, 0); crop_patch upsizes
        return [(0, 0)] * (per_side ** 2)
    pos = np.linspace(0, W - patch, per_side).astype(int)
    return [(int(x), int(y)) for y in pos for x in pos]


def patch_random(region, patch, n, rng):
    """n random top-left corners inside the region. Use for training."""
    W = region.size[0]
    if W <= patch:
        return [(0, 0)] * n
    xs = rng.integers(0, W - patch + 1, size=n)
    ys = rng.integers(0, W - patch + 1, size=n)
    return list(zip(xs.tolist(), ys.tolist()))


def crop_patch(region, xy, patch=128):
    x, y = xy
    p = region.crop((x, y, x + patch, y + patch))
    if p.size != (patch, patch):                       # region smaller than a patch: pad by resizing up
        p = p.resize((patch, patch), Image.BILINEAR)
    return p


# ---------------------------------------------------------------- augmentation
def augment(p, rng, patch=128):
    """Capture-variation augmentation for training patches. Same policy for both classes.

    Simulates: a different phone's colour science and exposure, mild defocus, sensor noise, in-app JPEG
    recompression, a different camera-to-subject distance, a slightly tilted phone. Never adds a grid.
    """
    if rng.random() < 0.5:
        p = ImageOps.mirror(p)
    if rng.random() < 0.5:                                  # distance: rescale then crop back
        f = rng.uniform(0.7, 1.4)
        s = max(patch, int(round(patch * f)))
        p = p.resize((s, s), Image.BILINEAR)
        x = rng.integers(0, s - patch + 1); y = rng.integers(0, s - patch + 1)
        p = p.crop((x, y, x + patch, y + patch))
    if rng.random() < 0.3:                                  # tilt
        p = p.rotate(rng.uniform(-5, 5), resample=Image.BILINEAR)
    p = ImageEnhance.Brightness(p).enhance(rng.uniform(0.8, 1.2))
    p = ImageEnhance.Contrast(p).enhance(rng.uniform(0.8, 1.2))
    p = ImageEnhance.Color(p).enhance(rng.uniform(0.8, 1.2))
    a = np.asarray(p).astype(np.float32)
    a *= rng.uniform(0.92, 1.08, size=3)[None, None, :]     # white balance shift
    if rng.random() < 0.5:                                  # sensor noise
        a += rng.normal(0, rng.uniform(1, 4), size=a.shape)
    p = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:                                  # mild defocus
        p = p.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
    if rng.random() < 0.5:                                  # JPEG recompression
        buf = io.BytesIO(); p.save(buf, "JPEG", quality=int(rng.integers(70, 96))); buf.seek(0)
        p = Image.open(buf).convert("RGB")
    return p


# ---------------------------------------------------------------- cache
class FaceCache:
    """Face boxes in faces.csv; native face regions as PNG under cache/faces/<cls>/<file>.

    Detection and the 37 MB attack PNGs are slow; every epoch reads the small cached region instead.
    """

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.csv = self.root / "derived" / "faces.csv"
        self.dir = self.root / "cache" / "faces"
        self.boxes = pd.read_csv(self.csv).set_index(["cls", "file"]) if self.csv.exists() else None

    def build(self, files, verbose=True):
        """files: iterable of (cls, filename). Detects, crops, caches. Idempotent."""
        rows = []
        for i, (cls, f) in enumerate(files):
            out = self.dir / cls / f
            im = load_image(DATA / cls / f)
            box = face_box(im)
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                face_region(im, box).save(out)
            rows.append(dict(cls=cls, file=f, cx=box[0], cy=box[1], side=box[2], found=box[3],
                             region_px=face_region(im, box).size[0]))
            if verbose and (i + 1) % 200 == 0:
                print(f"  {i + 1} cached")
        self.boxes = pd.DataFrame(rows).set_index(["cls", "file"])
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        self.boxes.to_csv(self.csv)
        return self.boxes

    def region(self, cls, f):
        return Image.open(self.dir / cls / f).convert("RGB")

    def box(self, cls, f):
        r = self.boxes.loc[(cls, f)]
        return float(r.cx), float(r.cy), float(r.side), bool(r.found)


if __name__ == "__main__":
    # smoke test: build the cache for every image, then draw a preview of patches
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = pd.read_csv(ROOT / "derived" / "splits.csv")
    fc = FaceCache()
    print("building face cache")
    boxes = fc.build(list(zip(splits.cls, splits.file)))
    print("faces found:", boxes.groupby(level=0).found.mean().round(3).to_dict())
    print("region px by class (median):", boxes.groupby(level=0).region_px.median().to_dict())

    rng = np.random.default_rng(0)
    examples = [("Bonafide", "01012.png"), ("Screens", "01012.png"), ("Bonafide", "10889.png"), ("Screens", "10889.png")]
    fig, ax = plt.subplots(len(examples), 9, figsize=(18, 2.2 * len(examples)))
    for r, (cls, f) in enumerate(examples):
        reg = fc.region(cls, f)
        ax[r, 0].imshow(reg); ax[r, 0].set_title(f"{cls[:4]} {f[:5]} region {reg.size[0]}px", fontsize=8); ax[r, 0].axis("off")
        grid = patch_grid(reg)
        assert len(grid) == 36
        for j, xy in enumerate(grid[:4]):
            ax[r, 1 + j].imshow(crop_patch(reg, xy)); ax[r, 1 + j].set_title("patch (eval grid)", fontsize=8); ax[r, 1 + j].axis("off")
        for j, xy in enumerate(patch_random(reg, 128, 4, rng)):
            ax[r, 5 + j].imshow(augment(crop_patch(reg, xy), rng)); ax[r, 5 + j].set_title("random + augment", fontsize=8); ax[r, 5 + j].axis("off")
    plt.tight_layout()
    (ROOT / "results").mkdir(exist_ok=True)
    plt.savefig(ROOT / "results" / "patches_preview.png", dpi=110)
    print("wrote results/patches_preview.png")
