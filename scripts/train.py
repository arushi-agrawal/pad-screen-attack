"""Patch CNN experiments. One script, one flag per experiment, same folds and metrics as the baseline.

    python scripts/train.py --aug none      # no augmentation
    python scripts/train.py --aug full      # full capture-variation augmentation (pad_data.augment)
    python scripts/train.py --aug full --protocol lopo          # phone transfer, both directions
    python scripts/train.py --aug full --eval density           # density-matched evaluation (attacks downsampled)

    python scripts/train.py --aug full --eval density --eval_only  # reuse the saved --aug full models, no retraining
    python scripts/train.py --aug full --resolution facenorm       # every face region resized to the bona fide median
                                                                    # (865 px) before patching, train and eval: density cue gone
    python scripts/train.py --aug wide                             # full augmentation plus random downscale 0.35x..1.0x of a
                                                                    # larger source crop: attacks seen at bona fide density in training
    python scripts/train.py --aug facerel                          # face-relative scale: every training patch covers a
                                                                    # fraction of the face drawn from ONE range for both classes
                                                                    # (6 % to 40 %), resized up or down to 128 px. Scale carries
                                                                    # no class information at all.
    python scripts/train.py --aug wide --protocol final           # FREEZE: train on all four development folds, score the
                                                                    # locked fold 4 exactly once, save artefacts/model_dev.pt
    python scripts/train.py --aug wide --protocol ship            # SHIP: identical recipe on all 1023 images, nothing held
                                                                    # out, save artefacts/model.pt (validated numbers belong to model_dev)

Model: ImageNet-pretrained MobileNetV3-small, first 7 feature blocks frozen (CPU budget: 35 patches/s vs 23
fully fine-tuned), binary head. 128 px patches at native
resolution from pad_data. Image score = mean of its 36 grid-patch probabilities (top-9 mean also stored).
Imbalance: attack images repeated ATTACK_REPEAT times per epoch so roughly a third of patches are attacks.
Development folds only; fold 4 never loaded except by --protocol final.
"""
import argparse, json, sys, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pad import data as P, metrics as M
from pad.model import make_model, to_tensor, PATCH, FACENORM_PX, MEAN, STD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
ATTACK_REPEAT = 20
WIDE_MIN, WIDE_MAX = 0.35, 1.0
FACEREL_MIN, FACEREL_MAX = 0.06, 0.40   # patch side as a fraction of the face width, both classes
torch.set_num_threads(4)


# augmentation policy per experiment; "wide" and "facerel" also change how the patch is cut (see TrainPatches)
AUG = {"none": lambda p, rng: p, "full": P.augment, "wide": P.augment, "facerel": P.augment}


class TrainPatches(Dataset):
    """Each item: one random patch from one image. Attack images are repeated so the epoch is roughly balanced."""

    def __init__(self, rows, cache, aug, patches_per_image, seed, facenorm=False, wide=False, facerel=False):
        self.facenorm, self.wide, self.facerel = facenorm, wide, facerel
        self.face_side = {(c, f): float(cache.boxes.loc[(c, f)].side) for c, f in zip(rows.cls, rows.file)}
        self.items = []
        for r in rows.itertuples():
            rep = ATTACK_REPEAT if r.label == 1 else 1
            self.items += [(r.cls, r.file, int(r.label))] * (rep * patches_per_image)
        self.cache, self.aug, self.rng = cache, aug, np.random.default_rng(seed)
        self.region = lru_cache(maxsize=300)(self._load_region)      # bounded: 8 GB machine

    def _load_region(self, cls, f):
        reg = self.cache.region(cls, f)
        return reg.resize((FACENORM_PX, FACENORM_PX), Image.LANCZOS) if self.facenorm else reg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cls, f, y = self.items[i]
        reg = self.region(cls, f)
        if self.facerel:
            # face-relative scale, identical distribution for both classes: the source crop covers a random
            # fraction of the face width and is resized to PATCH, up or down as needed
            c = self.rng.uniform(FACEREL_MIN, FACEREL_MAX)
            side = int(max(16, min(reg.size[0], round(c * self.face_side[(cls, f)]))))
            xy = P.patch_random(reg, side, 1, self.rng)[0]
            p = reg.crop((xy[0], xy[1], xy[0] + side, xy[1] + side)).resize((PATCH, PATCH), Image.LANCZOS)
        elif self.wide:
            # downscale by a random factor: take a larger source crop and shrink it to PATCH
            f_ = self.rng.uniform(WIDE_MIN, WIDE_MAX)
            side = int(min(reg.size[0], round(PATCH / f_)))
            xy = P.patch_random(reg, side, 1, self.rng)[0]
            p = reg.crop((xy[0], xy[1], xy[0] + side, xy[1] + side)).resize((PATCH, PATCH), Image.LANCZOS)
        else:
            xy = P.patch_random(reg, PATCH, 1, self.rng)[0]
            p = P.crop_patch(reg, xy)
        p = self.aug(p, self.rng)
        return to_tensor(p), torch.tensor(float(y))


def train(rows, cache, aug, epochs, patches_per_image, seed, facenorm=False, wide=False, facerel=False, log=print):
    torch.manual_seed(seed)
    ds = TrainPatches(rows, cache, aug, patches_per_image, seed, facenorm=facenorm, wide=wide, facerel=facerel)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    model = make_model(); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl))
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        log(f"      epoch {ep + 1}/{epochs}  loss {tot / n:.4f}  {time.time() - t0:.0f}s  ({len(ds)} patches)")
    model.eval()
    return model


@torch.no_grad()
def score_images(model, rows, cache, density_match=False, bona_region_px=None, facenorm=False):
    """Per-image scores from the 36 grid patches. density_match: shrink each region so its pixel density matches
    the median bona fide region (the direct test of the resolution confound)."""
    out = []
    for r in rows.itertuples():
        reg = cache.region(r.cls, r.file)
        if facenorm:
            reg = reg.resize((FACENORM_PX, FACENORM_PX), Image.LANCZOS)
        elif density_match and reg.size[0] > bona_region_px:
            s = int(round(bona_region_px))
            reg = reg.resize((s, s), Image.LANCZOS)
        grid = P.patch_grid(reg, PATCH)
        x = torch.stack([to_tensor(P.crop_patch(reg, xy, PATCH)) for xy in grid])
        pr = torch.sigmoid(model(x).squeeze(1)).numpy()
        out.append(dict(file=r.file, cls=r.cls, label=int(r.label), phone=r.phone, fold=int(r.fold),
                        score=float(pr.mean()), score_top9=float(np.sort(pr)[-9:].mean()), score_max=float(pr.max())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", choices=list(AUG), default="full")
    ap.add_argument("--protocol", choices=["devcv", "lopo", "final", "ship"], default="devcv")
    ap.add_argument("--eval", choices=["native", "density"], default="native")
    ap.add_argument("--resolution", choices=["native", "facenorm"], default="native")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patches_per_image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval_only", action="store_true", help="load saved fold models for this aug/protocol and only score")
    a = ap.parse_args()
    base = f"patchcnn_{a.aug}_{a.protocol}" + ("_facenorm" if a.resolution == "facenorm" else "")
    name = a.name or base + ("_density" if a.eval == "density" else "")
    ckpt_base = a.name or base          # checkpoints follow --name too, so a named run never overwrites the recipe's saved models
    model_path = lambda tag: OUT / f"{ckpt_base}_{tag.replace(' ', '_').replace(',', '')}.pt"

    splits = pd.read_csv(ROOT / "derived" / "splits.csv")
    dev = splits[~splits.locked].reset_index(drop=True)
    cache = P.FaceCache()
    bona_px = float(cache.boxes.loc["Bonafide"].region_px.median())
    print(f"{name}: dev set {len(dev)} images, {int(dev.label.sum())} attacks; fold 4 locked. aug={a.aug} eval={a.eval} resolution={a.resolution}")

    if a.protocol == "devcv":
        folds = [(dev[dev.fold != f], dev[dev.fold == f], f"fold {f}") for f in sorted(dev.fold.unique())]
    elif a.protocol == "final":
        locked = splits[splits.locked].reset_index(drop=True)
        folds = [(dev, locked, "final: train all dev, test locked fold 4")]
        print(f"   FINAL RUN. Locked fold 4 is being read for the first and only time: {len(locked)} images, {int(locked.label.sum())} attacks")
    elif a.protocol == "ship":
        everything = splits.reset_index(drop=True)
        folds = [(everything, dev, "ship: train on all 1023, in-sample check on dev")]
        print(f"   SHIP RUN. Training on all {len(everything)} images ({int(everything.label.sum())} attacks). The in-sample check below is optimistic by construction.")
    else:
        folds = []
        for held in ["Samsung", "iPhone"]:
            other = "iPhone" if held == "Samsung" else "Samsung"
            tr = dev[dev.lopo_group.isin([other, "bona_train"])]; va = dev[dev.lopo_group.isin([held, "bona_val"])]
            folds.append((tr, va, f"train {other}, test {held}"))

    results, all_scores = {}, []
    for tr, va, tag in folds:
        print(f"   {tag}: train {len(tr)} images ({int(tr.label.sum())} attacks), test {len(va)} ({int(va.label.sum())} attacks)")
        if a.eval_only:
            model = make_model(); model.load_state_dict(torch.load(model_path(tag))); model.eval()
            print(f"      loaded {model_path(tag).name}")
        else:
            model = train(tr, cache, AUG[a.aug], a.epochs, a.patches_per_image, a.seed, facenorm=(a.resolution == "facenorm"), wide=(a.aug == "wide"), facerel=(a.aug == "facerel"))
            torch.save(model.state_dict(), model_path(tag))
        t0 = time.time()
        sc = score_images(model, va, cache, density_match=(a.eval == "density"), bona_region_px=bona_px, facenorm=(a.resolution == "facenorm"))
        print(f"      scored {len(va)} images x 36 patches in {time.time() - t0:.0f}s")
        sc["split"] = tag; all_scores.append(sc)
        if a.protocol in ("lopo", "final"):
            res = M.summary(sc.score.values, sc.label.values, seed=a.seed)
            results[tag] = res; print(f"      {M.fmt(res)}")
        if a.protocol == "ship":
            art = ROOT / "artefacts"; art.mkdir(exist_ok=True)
            torch.save(model.state_dict(), art / "model.pt")
            cfg = json.load(open(art / "config.json")); cfg["trained_on"] = "all 1023 images (23 attacks); validated numbers belong to model_dev.pt"; cfg["validated_model"] = "model_dev.pt"
            json.dump(cfg, open(art / "config.json", "w"), indent=2)
            ea = pd.read_csv(OUT / "ea_scores.csv"); thr = cfg["thresholds"]
            for k, t in thr.items():
                att, bona = sc[sc.label == 1], sc[sc.label == 0]
                print(f"      in-sample dev at {k} ({t:.3f}): BPCER {(bona.score > t).mean():.3f}  APCER {(att.score <= t).mean():.3f}   (dev out-of-fold model_dev: BPCER {(ea[ea.label == 0].native > t).mean():.3f} APCER {(ea[ea.label == 1].native <= t).mean():.3f})")
            print(f"      saved artefacts/model.pt (shipped) and updated config.json; artefacts/model_dev.pt keeps the validated weights")
        if a.protocol == "final":
            art = ROOT / "artefacts"; art.mkdir(exist_ok=True)
            torch.save(model.state_dict(), art / "model_dev.pt")
            ea = pd.read_csv(OUT / "ea_scores.csv")           # out-of-fold dev scores of the chosen configuration
            thr = {f"bpcer_{int(b * 100)}pct": M.threshold_at_bpcer(ea.native, ea.label, b) for b in (0.01, 0.05, 0.10)}
            cfg = dict(model="mobilenet_v3_small", frozen_blocks=7, patch=PATCH, patches_per_image=36, aggregation="mean",
                       aug=a.aug, epochs=a.epochs, patches_per_image_train=a.patches_per_image, seed=a.seed,
                       trained_on="development folds 0-3 (820 images, 19 attacks)", score_direction="higher = attack",
                       thresholds_from="out-of-fold development scores (results/ea_scores.csv)", thresholds=thr,
                       imagenet_mean=MEAN.tolist(), imagenet_std=STD.tolist())
            json.dump(cfg, open(art / "config.json", "w"), indent=2)
            print(f"      saved artefacts/model.pt and artefacts/config.json  thresholds {thr}")
            # locked-fold scores at the dev thresholds, for the report
            for k, t in thr.items():
                att, bona = sc[sc.label == 1], sc[sc.label == 0]
                print(f"      fold 4 at dev threshold {k} ({t:.3f}): BPCER {(bona.score > t).mean():.3f}  APCER {(att.score <= t).mean():.3f}  (n_att={len(att)})")
                results[f"fold4_at_dev_{k}"] = dict(threshold=float(t), bpcer=float((bona.score > t).mean()), apcer=float((att.score <= t).mean()))

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(OUT / f"scores_{name}.csv", index=False)
    if a.protocol == "devcv":
        for col in ["score", "score_top9", "score_max"]:
            res = M.summary(scores[col].values, scores.label.values, seed=a.seed)
            results[col] = res
            print(f"   pooled dev-CV [{col}]  {M.fmt(res)}")
        thr = M.threshold_at_bpcer(scores.score.values, scores.label.values, 0.05)
        missed = scores[(scores.label == 1) & (scores.score <= thr)]
        results["missed_at_bpcer5"] = [f"{f[:5]} ({p})" for f, p in zip(missed.file, missed.phone)]
        print(f"   missed at BPCER 5 %: {len(missed)} of {int(scores.label.sum())}: {results['missed_at_bpcer5']}")
    results["config"] = vars(a)
    json.dump(results, open(OUT / f"{name}.json", "w"), indent=2, default=float)
    print(f"wrote results/{name}.json")


if __name__ == "__main__":
    main()
