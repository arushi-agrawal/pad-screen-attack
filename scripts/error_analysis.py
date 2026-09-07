"""Error analysis and cue attribution for the chosen model (patch CNN, wide augmentation).

Every development image is scored by the fold model that did not train on it (out-of-fold), so the analysis
is on honest scores. Fold 4 stays locked.

Two tools:
  1. Patch-score map      the 36 grid-patch probabilities drawn over the face region: where the model sees the screen.
  2. Cue ablation         re-score the image with one physical cue removed and record the drop:
        fine detail   Gaussian blur sigma 1.5 on the face region (kills grid, moire, fine noise; keeps colour)
        colour        grayscale (kills colour cast; keeps texture)
        density       region shrunk to the bona fide median 865 px (the step 7 density test)
        noise         5x5 median filter (kills sensor and re-capture noise; keeps edges and colour)
     The drop is how much of the image's attack score depended on that cue.

Outputs: results/error_analysis.json, results/ea_scores.csv (per image, per ablation), and three figures.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image, ImageFilter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pad import data as P, metrics as M
from pad.model import make_model, to_tensor, PATCH, FACENORM_PX

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
torch.set_num_threads(4)
MODEL = "patchcnn_wide_devcv"

splits = pd.read_csv(ROOT / "derived" / "splits.csv")
dev = splits[~splits.locked].reset_index(drop=True)
cache = P.FaceCache()
models = {}
for f in sorted(dev.fold.unique()):
    m = make_model(); m.load_state_dict(torch.load(OUT / f"{MODEL}_fold_{f}.pt")); m.eval(); models[f] = m

ABLATIONS = {
    "native": lambda reg: reg,
    "no_fine_detail": lambda reg: reg.filter(ImageFilter.GaussianBlur(1.5)),
    "no_colour": lambda reg: reg.convert("L").convert("RGB"),
    "density_matched": lambda reg: reg.resize((FACENORM_PX, FACENORM_PX), Image.LANCZOS) if reg.size[0] > FACENORM_PX else reg,
    "no_noise": lambda reg: Image.fromarray(cv2.medianBlur(np.asarray(reg), 5)),
}


@torch.no_grad()
def patch_probs(model, reg):
    grid = P.patch_grid(reg, PATCH)
    x = torch.stack([to_tensor(P.crop_patch(reg, xy, PATCH)) for xy in grid])
    return torch.sigmoid(model(x).squeeze(1)).numpy().reshape(6, 6)


rows, maps = [], {}
for i, r in enumerate(dev.itertuples()):
    reg = cache.region(r.cls, r.file); m = models[r.fold]
    rec = dict(file=r.file, cls=r.cls, label=int(r.label), phone=r.phone, fold=int(r.fold), region_px=reg.size[0])
    for name, fn in ABLATIONS.items():
        pm = patch_probs(m, fn(reg))
        rec[name] = float(pm.mean())
        if name == "native":
            maps[(r.cls, r.file)] = pm
    rows.append(rec)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(dev)}")
df = pd.DataFrame(rows)
for name in ABLATIONS:
    if name != "native":
        df[f"drop_{name}"] = df["native"] - df[name]
df.to_csv(OUT / "ea_scores.csv", index=False)

# ---------------- headline numbers ----------------
res = {}
thr5 = M.threshold_at_bpcer(df.native, df.label, 0.05); thr1 = M.threshold_at_bpcer(df.native, df.label, 0.01)
att, bona = df[df.label == 1].sort_values("native"), df[df.label == 0].sort_values("native", ascending=False)
res["threshold_bpcer5"], res["threshold_bpcer1"] = float(thr5), float(thr1)
res["attacks_missed_bpcer5"] = att[att.native <= thr5].file.tolist()
res["attacks_missed_bpcer1"] = att[att.native <= thr1].file.tolist()
res["lowest_scoring_attacks"] = att.head(5)[["file", "phone", "native"]].round(3).to_dict("records")
res["highest_scoring_bona_fide"] = bona.head(5)[["file", "native"]].round(3).to_dict("records")
res["score_summary"] = {"attack_median": float(att.native.median()), "attack_min": float(att.native.min()),
                        "bona_median": float(bona.native.median()), "bona_max": float(bona.native.max()),
                        "bona_99pct": float(bona.native.quantile(0.99))}
print("thresholds: BPCER5 %.3f  BPCER1 %.3f" % (thr5, thr1))
print("attack scores: min %.3f median %.3f | bona fide: median %.3f 99pct %.3f max %.3f" % (
    att.native.min(), att.native.median(), bona.native.median(), bona.native.quantile(.99), bona.native.max()))
print("missed at BPCER 5%:", res["attacks_missed_bpcer5"], " at BPCER 1%:", res["attacks_missed_bpcer1"])

# ---------------- cue attribution ----------------
cue_cols = [c for c in df.columns if c.startswith("drop_")]
by = df.groupby(["label", "phone"])[cue_cols].mean().round(3)
print("\nmean score drop when a cue is removed (attack score units):"); print(by)
res["cue_drop_by_class_phone"] = {f"{k[0]}|{k[1]}": v for k, v in by.to_dict("index").items()}
misses = df[(df.label == 1) & (df.density_matched <= M.threshold_at_bpcer(df.density_matched, df.label, 0.05))]
res["density_misses"] = misses[["file", "phone", "region_px", "native"] + list(ABLATIONS)[1:]].round(3).to_dict("records")
print("\nattacks missed under the density test, with their scores under each ablation:")
print(misses[["file", "phone", "region_px", "native", "no_fine_detail", "no_colour", "density_matched", "no_noise"]].round(3).to_string(index=False))

# ---------------- figure 1: cue attribution bars ----------------
fig, ax = plt.subplots(figsize=(9, 4))
labels = {"drop_no_fine_detail": "fine detail\n(blur)", "drop_no_colour": "colour\n(grayscale)", "drop_density_matched": "density\n(shrink to 865px)", "drop_no_noise": "noise\n(median filter)"}
groups = [("iPhone attacks", df[(df.label == 1) & (df.phone == "iPhone")]), ("Samsung attacks", df[(df.label == 1) & (df.phone == "Samsung")]), ("bona fide", df[df.label == 0])]
w = 0.25
for j, (name, g) in enumerate(groups):
    ax.bar(np.arange(len(cue_cols)) + (j - 1) * w, [g[c].mean() for c in cue_cols], w, label=name,
           yerr=[g[c].std() / np.sqrt(len(g)) for c in cue_cols], capsize=2)
ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(len(cue_cols))); ax.set_xticklabels([labels[c] for c in cue_cols])
ax.set_ylabel("drop in attack score when cue removed"); ax.set_title("What the model keys on: score drop per removed cue (out-of-fold, wide model)"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "ea_cue_attribution.png", dpi=120); plt.close()

# ---------------- figure 2: patch-score maps ----------------
def show_map(ax_img, ax_map, cls, f, title):
    reg = cache.region(cls, f); t = reg.copy(); t.thumbnail((300, 300))
    ax_img.imshow(t); ax_img.set_title(title, fontsize=8); ax_img.axis("off")
    im = ax_map.imshow(maps[(cls, f)], vmin=0, vmax=1, cmap="RdYlGn_r"); ax_map.set_title("patch scores (red = attack)", fontsize=8)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    for (yy, xx), v in np.ndenumerate(maps[(cls, f)]):
        ax_map.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=5, color="black")

sel = [("Screens", f, f"MISSED (density) {f[:5]} {p}") for f, p in zip(misses.file, misses.phone)]
sel += [("Screens", f, f"lowest native {f[:5]} {p}") for f, p in zip(att.head(2).file, att.head(2).phone)]
sel += [("Screens", f, f"easy attack {f[:5]} {p}") for f, p in zip(att.tail(2).file, att.tail(2).phone)]
sel += [("Bonafide", f, f"highest bona fide {f[:5]}") for f in bona.head(3).file]
n = len(sel); cols = 4
fig, axes = plt.subplots((n + cols - 1) // cols * 2, cols, figsize=(3.2 * cols, 3.4 * ((n + cols - 1) // cols) * 2 / 2 * 1.9))
for k, (cls, f, title) in enumerate(sel):
    rr, cc = (k // cols) * 2, k % cols
    show_map(axes[rr, cc], axes[rr + 1, cc], cls, f, title + f"  score {df[(df.cls == cls) & (df.file == f)].native.iloc[0]:.2f}")
for a in axes.flat:
    if not a.has_data(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "ea_patch_maps.png", dpi=110); plt.close()

# ---------------- figure 3: score distributions ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(bona.native, bins=40, alpha=.6, label="bona fide", density=True); ax[0].hist(att.native, bins=20, alpha=.6, label="attack", density=True)
ax[0].axvline(thr5, color="k", ls="--", lw=1, label="BPCER 5 % threshold"); ax[0].set(title="native resolution", xlabel="attack score"); ax[0].legend()
ax[1].hist(df[df.label == 0].density_matched, bins=40, alpha=.6, label="bona fide", density=True); ax[1].hist(df[df.label == 1].density_matched, bins=20, alpha=.6, label="attack", density=True)
ax[1].set(title="density-matched (attacks shrunk to 865 px)", xlabel="attack score"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "ea_score_distributions.png", dpi=120); plt.close()

json.dump(res, open(OUT / "error_analysis.json", "w"), indent=2, default=float)
print("\nwrote results/error_analysis.json, ea_scores.csv, ea_cue_attribution.png, ea_patch_maps.png, ea_score_distributions.png")
