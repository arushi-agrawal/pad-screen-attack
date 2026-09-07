"""Trade-off study for the chosen model (patch CNN, wide augmentation). No retraining.

  1. Operating point: real people flagged vs attacks passed at candidate thresholds, dev out-of-fold (820 images)
     with bootstrap bands, and the same thresholds applied to the locked fold as a spot check.
  2. Latency: where the time per image goes (decode, face detection, crop, patches, network), by input size.
  3. Time vs accuracy knobs, all scored out-of-fold with the four dev fold models:
        patches per image   4, 9, 16, 36
        input resolution    cap the face region at 512 / 865 / 1500 px or keep native
        aggregation         mean, median, top-9 mean, top-4 mean, max   (free: from the same 36 patch scores)
        threads             1, 2, 4                                     (time only)

Outputs: results/tradeoffs.json, results/to_patch_scores.npy, and figures results/to_*.png
"""
import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pad import data as P, metrics as M
from pad.model import make_model, to_tensor, PATCH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
torch.set_num_threads(4)

splits = pd.read_csv(ROOT / "derived" / "splits.csv")
dev = splits[~splits.locked].reset_index(drop=True)
cache = P.FaceCache()
models = {}
for f in sorted(dev.fold.unique()):
    m = make_model(); m.load_state_dict(torch.load(OUT / f"patchcnn_wide_devcv_fold_{f}.pt")); m.eval(); models[f] = m
res = {}

# =============================================================== 1. operating point
ea = pd.read_csv(OUT / "ea_scores.csv")                                # out-of-fold native scores, chosen model
final = pd.read_csv(OUT / "scores_patchcnn_wide_final.csv")           # locked fold, frozen model
s, y = ea.native.values, ea.label.values
n_att, n_bona = int(y.sum()), int((y == 0).sum())

def counts(scores, labels, thr):
    a, b = scores[labels == 1], scores[labels == 0]
    return dict(threshold=float(thr), real_flagged=int((b > thr).sum()), of_real=int(len(b)), bpcer=float((b > thr).mean()),
                attacks_passed=int((a <= thr).sum()), of_attacks=int(len(a)), apcer=float((a <= thr).mean()))

rows = []
for b in (0.005, 0.01, 0.02, 0.05, 0.10):
    thr = M.threshold_at_bpcer(s, y, b)
    d = counts(s, y, thr); d["set"] = "dev out-of-fold"; d["chosen_by"] = f"BPCER {b:.1%}"
    # bootstrap the attack-pass count at this fixed threshold
    rng = np.random.default_rng(0); a = s[y == 1]
    boots = [(rng.choice(a, len(a)) <= thr).mean() for _ in range(2000)]
    d["apcer_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    rows.append(d)
    d2 = counts(final.score.values, final.label.values, thr); d2["set"] = "locked fold 4"; d2["chosen_by"] = f"BPCER {b:.1%}"; rows.append(d2)
# also: the threshold that passes zero attacks on dev (attack-side operating point)
thr0 = float(s[y == 1].min()) - 1e-6
d = counts(s, y, thr0); d["set"] = "dev out-of-fold"; d["chosen_by"] = "zero attacks passed"; rows.append(d)
d2 = counts(final.score.values, final.label.values, thr0); d2["set"] = "locked fold 4"; d2["chosen_by"] = "zero attacks passed"; rows.append(d2)
op = pd.DataFrame(rows)
res["operating_points"] = op.to_dict("records")
print("1. Operating points (thresholds fixed on dev out-of-fold scores, then applied unchanged to the locked fold)")
print(op[["chosen_by", "set", "threshold", "real_flagged", "of_real", "attacks_passed", "of_attacks"]].round(3).to_string(index=False))

# DET-style curve with bootstrap band
def curve(scores, labels, grid):
    return np.array([[(scores[labels == 0] > t).mean(), (scores[labels == 1] <= t).mean()] for t in grid])
grid = np.linspace(0, 1, 401); c = curve(s, y, grid)
rng = np.random.default_rng(0); B = []
for _ in range(500):
    ia = rng.choice(np.where(y == 1)[0], n_att); ib = rng.choice(np.where(y == 0)[0], n_bona); idx = np.r_[ia, ib]
    B.append(curve(s[idx], y[idx], grid))
B = np.array(B); lo, hi = np.percentile(B[:, :, 1], 2.5, axis=0), np.percentile(B[:, :, 1], 97.5, axis=0)
fig, ax = plt.subplots(figsize=(7, 5))
ax.fill_betweenx(c[:, 0], lo, hi, alpha=.2, label="95 % bootstrap band")
ax.plot(c[:, 1], c[:, 0], lw=2, label="dev out-of-fold (19 attacks, 801 bona fide)")
cf = curve(final.score.values, final.label.values, grid); ax.plot(cf[:, 1], cf[:, 0], lw=1.5, ls="--", label="locked fold 4 (4 attacks, 199 bona fide)")
for b in (0.01, 0.05, 0.10):
    t = M.threshold_at_bpcer(s, y, b); ax.scatter([(s[y == 1] <= t).mean()], [b], color="k", zorder=5); ax.annotate(f"BPCER {b:.0%}, thr {t:.2f}", ((s[y == 1] <= t).mean() + .01, b + .005), fontsize=8)
ax.set(xlabel="attacks passed (APCER)", ylabel="real people flagged (BPCER)", xlim=(-.01, .6), ylim=(-.005, .2), title="Operating curve of the chosen model"); ax.legend(loc="upper right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "to_operating_curve.png", dpi=120); plt.close()

# =============================================================== 2. latency breakdown by input size
print("\n2. Latency breakdown per image (seconds), by input size")
model = models[0]
def timed_pipeline(path):
    t = {}; t0 = time.perf_counter()
    im = P.load_image(path); t["decode"] = time.perf_counter() - t0; t0 = time.perf_counter()
    box = P.face_box(im); t["face_detect"] = time.perf_counter() - t0; t0 = time.perf_counter()
    reg = P.face_region(im, box); grid = P.patch_grid(reg, PATCH)
    x = torch.stack([to_tensor(P.crop_patch(reg, xy, PATCH)) for xy in grid]); t["crop_and_patches"] = time.perf_counter() - t0; t0 = time.perf_counter()
    with torch.no_grad(): torch.sigmoid(model(x).squeeze(1)); t["network_36_patches"] = time.perf_counter() - t0
    t["total"] = sum(t.values()); t["megapixels"] = im.size[0] * im.size[1] / 1e6
    return t
sizes = {"1024 px (bona fide)": dev[dev.label == 0].head(10), "1700-2200 px (attack)": dev[(dev.label == 1)].copy()}
att_files = dev[dev.label == 1].merge(cache.boxes.reset_index()[["cls", "file", "region_px"]], on=["cls", "file"])
img_px = {r.file: Image.open(P.DATA / "Screens" / r.file).size[0] for r in att_files.itertuples()}
att_files["img_px"] = att_files.file.map(img_px)
buckets = {"1024 px, 1 MP (bona fide)": [("Bonafide", f) for f in dev[dev.label == 0].head(8).file],
           "1700-2700 px, 3-7 MP (attack)": [("Screens", f) for f in att_files[att_files.img_px < 2800].head(6).file],
           "2800-4300 px, 8-16 MP (attack)": [("Screens", f) for f in att_files[(att_files.img_px >= 2800) & (att_files.img_px < 5000)].head(6).file],
           "5712 px, 24 MP (attack)": [("Screens", f) for f in att_files[att_files.img_px >= 5000].file]}
lat = []
for name, files in buckets.items():
    ts = [timed_pipeline(P.DATA / c / f) for c, f in files]
    row = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}; row["bucket"] = name; row["n"] = len(ts); lat.append(row)
lat = pd.DataFrame(lat).set_index("bucket")
print(lat[["megapixels", "decode", "face_detect", "crop_and_patches", "network_36_patches", "total"]].round(3).to_string())
res["latency_by_size"] = lat.reset_index().to_dict("records")
fig, ax = plt.subplots(figsize=(9, 3.8))
lat[["decode", "face_detect", "crop_and_patches", "network_36_patches"]].plot.barh(stacked=True, ax=ax)
ax.set(xlabel="seconds per image (median)", title="Where the time goes, by input size (4 CPU threads)"); ax.invert_yaxis(); plt.tight_layout(); plt.savefig(OUT / "to_latency.png", dpi=120); plt.close()

# threads
print("\n   network time for 36 patches vs threads:")
x = torch.randn(36, 3, PATCH, PATCH); thr_rows = {}
for nt in (1, 2, 4):
    torch.set_num_threads(nt)
    with torch.no_grad():
        model(x); t0 = time.perf_counter()
        for _ in range(5): model(x)
    thr_rows[nt] = (time.perf_counter() - t0) / 5; print(f"     {nt} thread(s): {thr_rows[nt]:.3f} s")
torch.set_num_threads(4); res["network_seconds_by_threads"] = {str(k): float(v) for k, v in thr_rows.items()}

# =============================================================== 3. time vs accuracy knobs (out-of-fold)
print("\n3. Time vs accuracy, out-of-fold on the 820 development images")

@torch.no_grad()
def oof_patch_scores(per_side=6, cap=None):
    """Per-image patch-score arrays, each image scored by its held-out fold model. cap: max region px."""
    out = np.zeros((len(dev), per_side * per_side), np.float32); t0 = time.perf_counter()
    for i, r in enumerate(dev.itertuples()):
        reg = cache.region(r.cls, r.file)
        if cap and reg.size[0] > cap:
            reg = reg.resize((cap, cap), Image.LANCZOS)
        grid = P.patch_grid(reg, PATCH, per_side=per_side)
        x = torch.stack([to_tensor(P.crop_patch(reg, xy, PATCH)) for xy in grid])
        out[i] = torch.sigmoid(models[r.fold](x).squeeze(1)).numpy()
    return out, (time.perf_counter() - t0) / len(dev)

def evaluate(scores, label):
    r = M.summary(scores, y, n_boot=500)
    thr = M.threshold_at_bpcer(scores, y, 0.05)
    return dict(config=label, auc=round(r["auc"], 4), auc_lo=round(r["auc_ci"][0], 3), apcer5=round(r["apcer@bpcer5"], 3),
                attacks_passed_at_bpcer5=int((scores[y == 1] <= thr).sum()), apcer1=round(r["apcer@bpcer1"], 3))

knob_rows = []
P36, t36 = oof_patch_scores(6)
np.save(OUT / "to_patch_scores.npy", P36)
print("   patches per image:")
for ps in (2, 3, 4, 6):
    if ps == 6: pm, tt = P36, t36
    else: pm, tt = oof_patch_scores(ps)
    d = evaluate(pm.mean(1), f"{ps * ps} patches, native, mean"); d["knob"] = "patches"; d["seconds_per_image"] = round(tt, 3); knob_rows.append(d)
    print(f"     {ps * ps:2d} patches: AUC {d['auc']:.3f}  attacks passed @5% {d['attacks_passed_at_bpcer5']}/19  {tt:.2f}s/img")
print("   input resolution cap (region resized down to at most N px; below N left alone):")
for cap in (512, 865, 1500):
    pm, tt = oof_patch_scores(6, cap=cap)
    d = evaluate(pm.mean(1), f"36 patches, cap {cap} px, mean"); d["knob"] = "resolution"; d["seconds_per_image"] = round(tt, 3); knob_rows.append(d)
    print(f"     cap {cap:4d} px: AUC {d['auc']:.3f}  attacks passed @5% {d['attacks_passed_at_bpcer5']}/19  {tt:.2f}s/img")
print("   aggregation of the 36 native patch scores:")
aggs = {"mean": P36.mean(1), "median": np.median(P36, 1), "top-9 mean": np.sort(P36, 1)[:, -9:].mean(1), "top-4 mean": np.sort(P36, 1)[:, -4:].mean(1), "max": P36.max(1)}
for name, sc in aggs.items():
    d = evaluate(sc, f"36 patches, native, {name}"); d["knob"] = "aggregation"; d["seconds_per_image"] = round(t36, 3)
    d["real_flagged_at_1pct_thr"] = int((sc[y == 0] > M.threshold_at_bpcer(sc, y, 0.01)).sum())
    knob_rows.append(d); print(f"     {name:<11s}: AUC {d['auc']:.3f}  attacks passed @5% {d['attacks_passed_at_bpcer5']}/19  @1% {d['apcer1']:.2f}")
knobs = pd.DataFrame(knob_rows); res["knobs"] = knob_rows

fig, ax = plt.subplots(figsize=(8, 4.5))
for knob, mk in (("patches", "o"), ("resolution", "s")):
    g = knobs[knobs.knob == knob]; ax.plot(g.seconds_per_image, g.auc, marker=mk, label=knob)
    for r in g.itertuples(): ax.annotate(r.config.split(",")[0] if knob == "patches" else r.config.split(",")[1].strip(), (r.seconds_per_image, r.auc), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set(xlabel="seconds per image (dev set, 4 threads, cached face region)", ylabel="AUC (out-of-fold)", title="Time vs accuracy: patches per image and input resolution cap"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "to_time_vs_accuracy.png", dpi=120); plt.close()

json.dump(res, open(OUT / "tradeoffs.json", "w"), indent=2, default=float)
print("\nwrote results/tradeoffs.json, to_operating_curve.png, to_latency.png, to_time_vs_accuracy.png")
