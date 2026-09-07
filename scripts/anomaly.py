"""Anomaly detection: the no-attack-labels reference. Trained on bona fide only; attacks never seen in training.

Two feature spaces, same protocol as the baseline (dev folds, fold 4 locked, phone transfer check):
  classical   the six baseline features (LBP distance measured to the training bona fide mean)
  embedding   ImageNet MobileNetV3-small penultimate features of the 256 px whole-face view, and of the mean of
              the 36 native-resolution patches (both 576-d)

Scorer: Gaussian density on the training bona fide (mean + shrunk covariance); score = Mahalanobis distance.
Higher = more unusual = more likely attack. Everything a one-class method can do with these features.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import torch, torchvision
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from pad import data as P, metrics as M
from pad.model import to_tensor, PATCH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)

splits = pd.read_csv(ROOT / "derived" / "splits.csv")
dev = splits[~splits.locked].reset_index(drop=True)
cache = P.FaceCache()

# ---------------- feature spaces ----------------
feat = pd.read_csv(ROOT / "derived" / "classical_features.csv")
LBP_G = np.load(ROOT / "derived" / "lbp_gray.npy")
SCALAR = ["lap_var", "cr_std", "hf_energy", "peakiness_native", "specular_ratio"]   # plus LBP distance below: the six baseline features
fidx = {(c, f): i for i, (c, f) in enumerate(zip(feat.cls, feat.file))}
dev["frow"] = [fidx[(c, f)] for c, f in zip(dev.cls, dev.file)]


def chi2(H, m):
    return 0.5 * np.sum((H - m) ** 2 / (H + m + 1e-12), axis=1)


def classical(train_idx, apply_idx):
    tb = dev.frow.values[train_idx][dev.label.values[train_idx] == 0]
    mg = LBP_G[tb].mean(0)
    rows = dev.frow.values[apply_idx]
    return np.column_stack([feat.loc[rows, SCALAR].to_numpy(float), chi2(LBP_G[rows], mg)])


EMB_CACHE = OUT / "embeddings_mnv3.npz"
if EMB_CACHE.exists():
    z = np.load(EMB_CACHE); EMB_FACE, EMB_PATCH = z["face"], z["patch"]
else:
    print("extracting embeddings (once)")
    m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    m.classifier = torch.nn.Identity()
    EMB_FACE, EMB_PATCH = np.zeros((len(dev), 576), np.float32), np.zeros((len(dev), 576), np.float32)
    with torch.no_grad():
        for i, r in enumerate(dev.itertuples()):
            reg = cache.region(r.cls, r.file)
            box = cache.box(r.cls, r.file)
            fv = P.face_view(P.load_image(P.DATA / r.cls / r.file), box, 256)
            EMB_FACE[i] = m(to_tensor(fv)[None]).numpy()[0]
            x = torch.stack([to_tensor(P.crop_patch(reg, xy, PATCH)) for xy in P.patch_grid(reg, PATCH)])
            EMB_PATCH[i] = m(x).numpy().mean(0)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(dev)}")
    np.savez(EMB_CACHE, face=EMB_FACE, patch=EMB_PATCH)

SPACES = {
    "classical (6)": classical,
    "embedding, whole face 256px": lambda tr, ap: EMB_FACE[ap],
    "embedding, mean of 36 native patches": lambda tr, ap: EMB_PATCH[ap],
}


def gaussian_scores(Xtr_bona, Xap):
    sc = StandardScaler().fit(Xtr_bona)
    Z, Za = sc.transform(Xtr_bona), sc.transform(Xap)
    lw = LedoitWolf().fit(Z)
    return lw.mahalanobis(Za)


def run(name, folds, featurise):
    oof = np.full(len(dev), np.nan)
    for tr, va in folds:
        Xtr = featurise(tr, tr)[dev.label.values[tr] == 0]        # bona fide only: attacks never seen
        oof[va] = gaussian_scores(Xtr, featurise(tr, va))
    mask = ~np.isnan(oof)
    res = M.summary(oof[mask], dev.label.values[mask], seed=0)
    print(f"  {name:<58s} {M.fmt(res)}")
    return oof, res


devcv = [(np.where(dev.fold != f)[0], np.where(dev.fold == f)[0]) for f in sorted(dev.fold.unique())]
def lopo(held):
    other = "Samsung" if held == "iPhone" else "iPhone"
    return [(np.where(dev.lopo_group.isin([other, "bona_train"]))[0], np.where(dev.lopo_group.isin([held, "bona_val"]))[0])]

results = {}
print("Anomaly detection (Gaussian on training bona fide; attacks never seen)")
for sp, fn in SPACES.items():
    oof, res = run(f"{sp} / dev-CV", devcv, fn); results[f"{sp}/devcv"] = res
    pd.DataFrame({"file": dev.file, "label": dev.label, "phone": dev.phone, "fold": dev.fold, "score": oof}).to_csv(OUT / f"oof_anomaly_{sp.split(',')[0].split(' ')[0]}_{'patch' if 'patch' in sp else 'face' if 'face' in sp else 'classical'}.csv", index=False)
    for held in ["Samsung", "iPhone"]:
        _, res = run(f"{sp} / test {held}", lopo(held), fn); results[f"{sp}/lopo_test_{held}"] = res
json.dump(results, open(OUT / "anomaly.json", "w"), indent=2, default=float)
print("wrote results/anomaly.json")
