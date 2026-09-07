"""Build the fold assignments every later experiment reads.

Writes derived/splits.csv (one row per image) and derived/splits.json (the protocol and seed).
Run once. Never re-roll without bumping SEED and saying so in the decision log.

Columns in splits.csv
  file        filename, e.g. 01012.png
  cls         Bonafide | Screens
  label       0 = bona fide, 1 = attack
  pair_id     filename stem; an attack and its bona fide twin share it
  paired      True if this bona fide image has an attack twin (or is an attack)
  phone       capture phone for attacks (iPhone | Samsung); "none" for bona fide
  fold        0..4  primary protocol: stratified 5-fold, grouped by pair, attacks stratified by phone
  lopo_group  leave-one-phone-out protocol: iPhone | Samsung for paired images,
              bona_train | bona_val for unpaired bona fide (fixed 20 % val subset)
  random_split  train | val  naive 80/20 stratified by class only, pairs ignored (for the report's comparison)
  locked      True for fold 4. Locked fold: not used for anything during development. Folds 0..3 are the
              development set (4-fold cross-validation). Fold 4 is run exactly once, on the frozen final model,
              as a check that the development process did not fool itself.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from PIL import Image
from pad import data as P

SEED = 20260905
N_FOLDS = 5
LOCKED_FOLD = 4
LOPO_BONA_VAL_FRAC = 0.20
RANDOM_VAL_FRAC = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA = P.DATA
rng = np.random.default_rng(SEED)

rows = []
for cls, label in (("Bonafide", 0), ("Screens", 1)):
    for p in sorted((DATA / cls).glob("*.png")):
        rows.append(dict(file=p.name, cls=cls, label=label, pair_id=p.stem))
df = pd.DataFrame(rows)

attack_ids = set(df.loc[df.label == 1, "pair_id"])
df["paired"] = df.pair_id.isin(attack_ids)

def phone_of(name):
    make = Image.open(DATA / "Screens" / name).getexif().get(0x010F, "")
    return "iPhone" if make == "Apple" else "Samsung" if make.lower() == "samsung" else "unknown"
phone_by_pair = {n[:-4]: phone_of(n) for n in df.loc[df.label == 1, "file"]}
df["phone"] = df.apply(lambda r: phone_by_pair.get(r.pair_id, "none") if r.paired else "none", axis=1)

# ---- primary: 5-fold, grouped by pair, attacks stratified by phone ----
fold_of_pair = {}
for phone in sorted(set(phone_by_pair.values())):
    ids = sorted(k for k, v in phone_by_pair.items() if v == phone)
    rng.shuffle(ids)
    # spread this phone's pairs across folds; start at the fold with the fewest attacks so far
    counts = np.zeros(N_FOLDS, int)
    for k, v in fold_of_pair.items():
        counts[v] += 1
    start = int(np.argmin(counts))
    for i, pid in enumerate(ids):
        fold_of_pair[pid] = (start + i) % N_FOLDS

unpaired = sorted(df.loc[~df.paired, "pair_id"])
rng.shuffle(unpaired)
for i, pid in enumerate(unpaired):
    fold_of_pair[pid] = i % N_FOLDS
df["fold"] = df.pair_id.map(fold_of_pair)
df["locked"] = df.fold == LOCKED_FOLD

# ---- leave-one-phone-out ----
val_bona = set(rng.choice(unpaired, size=int(round(LOPO_BONA_VAL_FRAC * len(unpaired))), replace=False))
def lopo(r):
    if r.paired:
        return phone_by_pair[r.pair_id]
    return "bona_val" if r.pair_id in val_bona else "bona_train"
df["lopo_group"] = df.apply(lopo, axis=1)

# ---- naive random 80/20 (what not to do) ----
df["random_split"] = "train"
for label in (0, 1):
    idx = df.index[df.label == label].to_numpy()
    val = rng.choice(idx, size=int(round(RANDOM_VAL_FRAC * len(idx))), replace=False)
    df.loc[val, "random_split"] = "val"

df = df.sort_values(["label", "file"]).reset_index(drop=True)
(ROOT / "derived").mkdir(exist_ok=True)
df.to_csv(ROOT / "derived" / "splits.csv", index=False)

# ---- integrity checks ----
g = df.groupby("pair_id")
assert (g.fold.nunique() == 1).all(), "a pair is split across folds"
assert (g.lopo_group.nunique() == 1).all(), "a pair is split across lopo groups"
assert df.fold.between(0, N_FOLDS - 1).all()
assert set(df.loc[df.label == 1, "phone"]) <= {"iPhone", "Samsung"}

def d2s(d):
    return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): int(v) for k, v in d.items()}

summary = {
    "seed": SEED,
    "n_images": int(len(df)),
    "n_attacks": int(df.label.sum()),
    "primary": {
        "protocol": f"stratified {N_FOLDS}-fold, grouped by pair, attacks stratified by phone; every attack validated once",
        "attacks_per_fold": d2s(df[df.label == 1].groupby("fold").size().to_dict()),
        "attacks_per_fold_by_phone": d2s(df[df.label == 1].groupby(["fold", "phone"]).size().to_dict()),
        "bona_fide_per_fold": d2s(df[df.label == 0].groupby("fold").size().to_dict()),
    },
    "leave_one_phone_out": {
        "protocol": "hold out all attacks (and twins) from one phone plus a fixed 20 % of unpaired bona fide; train on the rest; run both directions",
        "groups": d2s(df.groupby(["lopo_group", "label"]).size().to_dict()),
    },
    "random_split": {
        "protocol": "80/20 stratified by class only, pairs and phones ignored; for comparison in the report only",
        "counts": d2s(df.groupby(["random_split", "label"]).size().to_dict()),
    },
    "locked_fold": LOCKED_FOLD,
    "rules": [
        f"fold {LOCKED_FOLD} is locked: not used for training, validation, selection, or inspection during development; run once on the frozen final model",
        f"development uses folds 0..{LOCKED_FOLD - 1} as {LOCKED_FOLD}-fold cross-validation",
        "pairs always share a fold and a lopo group",
        "any model selection, threshold, or hyperparameter choice happens inside the training folds",
        "augmentation and resampling (SMOTE, oversampling, synthetic attacks) happen after the split, inside the training fold only; validation folds are never resampled",
        "the primary 5-fold number is never used to pick anything",
    ],
}
(ROOT / "derived" / "splits.json").write_text(json.dumps(summary, indent=2))

print(f"wrote splits.csv ({len(df)} rows) and splits.json  seed={SEED}")
print("\nattacks per fold by phone:")
print(df[df.label == 1].pivot_table(index="fold", columns="phone", values="file", aggfunc="count", fill_value=0).assign(total=lambda t: t.sum(axis=1)))
print("\nbona fide per fold:", df[df.label == 0].groupby("fold").size().to_dict())
print("\nleave-one-phone-out groups:")
print(df.groupby(["lopo_group", "label"]).size().unstack(fill_value=0))
print("\nrandom 80/20:")
print(df.groupby(["random_split", "label"]).size().unstack(fill_value=0))
print("\nintegrity: pairs intact in fold and lopo_group; OK")
