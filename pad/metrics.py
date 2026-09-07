"""PAD metrics (ISO/IEC 30107-3 vocabulary). Score convention: higher = more likely attack.

  apcer_at_bpcer(scores, labels, bpcer)  attack miss rate when bona fide rejection is capped at `bpcer`
  eer(scores, labels)                    equal error rate
  auc(scores, labels)                    ROC AUC
  summary(scores, labels, ...)           all of the above with bootstrap confidence intervals
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _split(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    return scores[labels == 1], scores[labels == 0]


def threshold_at_bpcer(scores, labels, bpcer):
    """Highest threshold such that the fraction of bona fide scored above it is <= bpcer."""
    _, bona = _split(scores, labels)
    return float(np.quantile(bona, 1.0 - bpcer, method="higher"))


def apcer_at_bpcer(scores, labels, bpcer):
    att, _ = _split(scores, labels)
    thr = threshold_at_bpcer(scores, labels, bpcer)
    return float(np.mean(att <= thr))


def bpcer_at_threshold(scores, labels, thr):
    _, bona = _split(scores, labels)
    return float(np.mean(bona > thr))


def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def auc(scores, labels):
    return float(roc_auc_score(labels, scores))


def summary(scores, labels, bpcers=(0.01, 0.05), n_boot=2000, seed=0):
    """Point estimates plus stratified bootstrap 95 % intervals. Returns a flat dict."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    att_idx, bona_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)

    def all_metrics(s, l):
        out = {"auc": auc(s, l), "eer": eer(s, l)}
        for b in bpcers:
            out[f"apcer@bpcer{int(b * 100)}"] = apcer_at_bpcer(s, l, b)
        return out

    point = all_metrics(scores, labels)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        ia = rng.choice(att_idx, len(att_idx), replace=True)
        ib = rng.choice(bona_idx, len(bona_idx), replace=True)
        idx = np.concatenate([ia, ib])
        try:
            m = all_metrics(scores[idx], labels[idx])
        except ValueError:
            continue
        for k, v in m.items():
            boots[k].append(v)
    out = {"n_attacks": int(len(att_idx)), "n_bona": int(len(bona_idx))}
    for k, v in point.items():
        lo, hi = (np.percentile(boots[k], [2.5, 97.5]) if boots[k] else (np.nan, np.nan))
        out[k] = v
        out[f"{k}_ci"] = (float(lo), float(hi))
    return out


def fmt(res):
    """One-line human summary of a summary() dict."""
    def ci(k):
        lo, hi = res[f"{k}_ci"]
        return f"{res[k]:.3f} [{lo:.3f}, {hi:.3f}]"
    return (f"AUC {ci('auc')}  EER {ci('eer')}  "
            f"APCER@1% {ci('apcer@bpcer1')}  APCER@5% {ci('apcer@bpcer5')}  "
            f"(n_att={res['n_attacks']}, n_bona={res['n_bona']})")
