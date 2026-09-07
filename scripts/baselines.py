"""Classical baseline on the fixed splits. Development folds only (fold 4 stays locked).

One baseline: the classical model. Six hand-crafted features, one per physical cue, (computed on the face-normalised
512 px view, so image size, file size and metadata never enter) into a logistic regression. This is the bar a
learned model has to beat.

Protocols (from splits.csv):
  dev-CV     4-fold cross-validation on folds 0..3, grouped by pair, C chosen inside the training folds
  LOPO       train on one phone's attacks, test on the other's. The closest thing we have to the hidden set.

Outputs: results/baselines.json, results/oof_<name>.csv, printed tables.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from pad import metrics as M

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
SEED = 0

splits = pd.read_csv(ROOT / "derived" / "splits.csv")
feat = pd.read_csv(ROOT / "derived" / "classical_features.csv")
LBP_G = np.load(ROOT / "derived" / "lbp_gray.npy")

df = splits.merge(feat.drop(columns=["lbp_dist_gray"], errors="ignore"), on=["file", "cls"])
assert len(df) == 1023
lbp_index = {(c, f): i for i, (c, f) in enumerate(zip(feat.cls, feat.file))}
df["lbp_row"] = [lbp_index[(c, f)] for c, f in zip(df.cls, df.file)]

dev = df[~df.locked].reset_index(drop=True)
print(f"development set: {len(dev)} images, {int(dev.label.sum())} attacks; fold 4 locked and untouched\n")

# one feature per physical cue the literature uses: blur, colour cast, fine energy, moire, reflection, surface texture
SCALAR = ["lap_var", "cr_std", "hf_energy", "peakiness_native", "specular_ratio"]
LBP_COLS = ["lbp_dist_gray"]


def chi2(H, m):
    return 0.5 * np.sum((H - m) ** 2 / (H + m + 1e-12), axis=1)


def classical_matrix(train_idx, apply_idx):
    """Scalar features plus LBP distances to the *training-fold* bona fide mean (no validation leakage)."""
    tb = train_idx[dev.label.values[train_idx] == 0]
    mg = LBP_G[dev.lbp_row.values[tb]].mean(0)
    X = dev.loc[apply_idx, SCALAR].to_numpy(float)
    rows = dev.lbp_row.values[apply_idx]
    return np.column_stack([X, chi2(LBP_G[rows], mg)])


def logreg(X, y, groups):
    """Logistic regression on standardised features; C chosen by 3-fold CV inside the training data."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED))
    gs = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc", cv=inner.split(X, y, groups), n_jobs=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_["logisticregression__C"]


def run_protocol(name, folds, featurise, verbose=True):
    """folds: list of (train_idx, val_idx). Returns pooled out-of-fold scores and per-fold notes."""
    oof = np.full(len(dev), np.nan); notes = []
    for k, (tr, va) in enumerate(folds):
        Xtr, Xva = featurise(tr, tr), featurise(tr, va)
        ytr = dev.label.values[tr]
        model, C = logreg(Xtr, ytr, dev.pair_id.values[tr])
        oof[va] = model.predict_proba(Xva)[:, 1]
        notes.append({"fold": k, "n_train_att": int(ytr.sum()), "n_val_att": int(dev.label.values[va].sum()), "C": C})
    mask = ~np.isnan(oof)
    res = M.summary(oof[mask], dev.label.values[mask], seed=SEED)
    res["folds"] = notes
    if verbose:
        print(f"  {name:<34s} {M.fmt(res)}")
    return oof, res


def dev_cv_folds():
    return [(np.where(dev.fold != f)[0], np.where(dev.fold == f)[0]) for f in sorted(dev.fold.unique())]


def lopo_folds(held):
    other = "Samsung" if held == "iPhone" else "iPhone"
    tr = np.where(dev.lopo_group.isin([other, "bona_train"]))[0]
    va = np.where(dev.lopo_group.isin([held, "bona_val"]))[0]
    return [(tr, va)]


results = {}

# ---------- classical model ----------
print(f"Classical model: {len(SCALAR) + len(LBP_COLS)} hand-crafted features into logistic regression")
oof, res = run_protocol("dev-CV (the bar to beat)", dev_cv_folds(), classical_matrix)
results["classical_devcv"] = res
pd.DataFrame({"file": dev.file, "label": dev.label, "phone": dev.phone, "fold": dev.fold, "score": oof}).to_csv(OUT / "oof_classical_devcv.csv", index=False)
thr = M.threshold_at_bpcer(oof, dev.label.values, 0.05)
missed = dev.loc[(dev.label == 1) & (oof <= thr), ["file", "phone"]]
results["classical_devcv"]["missed_at_bpcer5"] = [f"{f[:5]} ({p})" for f, p in zip(missed.file, missed.phone)]
print(f"    missed at BPCER 5 %: {len(missed)} of {int(dev.label.sum())}: {results['classical_devcv']['missed_at_bpcer5']}")

print("\n   Phone transfer check (same model, train on one phone, test on the other)")
for held in ["Samsung", "iPhone"]:
    oof, res = run_protocol(f"train {'iPhone' if held == 'Samsung' else 'Samsung'}, test {held}", lopo_folds(held), classical_matrix)
    results[f"classical_lopo_test_{held}"] = res

# ---------- what the classical model keys on ----------
tr = np.arange(len(dev))
model, C = logreg(classical_matrix(tr, tr), dev.label.values, dev.pair_id.values)
coef = model.named_steps["logisticregression"].coef_[0]
names = SCALAR + LBP_COLS
weights = sorted(zip(names, coef), key=lambda t: -abs(t[1]))
print("\n   What it keys on (standardised weights, top 5; + means higher value -> attack):")
for n, w in weights[:5]:
    print(f"     {n:<18s} {w:+.2f}")
results["classical_weights_devfit"] = {n: float(w) for n, w in weights}

json.dump(results, open(OUT / "baselines.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT / 'baselines.json'} and out-of-fold score files")
