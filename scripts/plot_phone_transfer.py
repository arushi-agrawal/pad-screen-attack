"""Unseen-camera bar plot for the report: original size (left) and attacks shrunk (right). Re-run after any lopo experiment."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
R=Path(__file__).resolve().parents[1]/"results"; OUT=R/"fig_phone_transfer.png"
J=lambda n: json.load(open(R/f"{n}.json")) if (R/f"{n}.json").exists() else None
tS,tI="train iPhone, test Samsung","train Samsung, test iPhone"
b=J("baselines")
panels=[("Original size", [("Classical\nbaseline",b["classical_lopo_test_Samsung"],b["classical_lopo_test_iPhone"]),
                          ("Every face\nresized to 865 px",J("patchcnn_full_lopo_facenorm")[tS],J("patchcnn_full_lopo_facenorm")[tI]),
                          ("Face-relative\nscale range",J("patchcnn_facerel_lopo")[tS],J("patchcnn_facerel_lopo")[tI]),
                          ("Capture\naugmentation",J("patchcnn_full_lopo")[tS],J("patchcnn_full_lopo")[tI]),
                          ("Final model",J("patchcnn_wide_lopo")[tS],J("patchcnn_wide_lopo")[tI])]),
        ("Attacks shrunk to bona fide pixel density", [("Capture\naugmentation",J("patchcnn_full_lopo_density")[tS],J("patchcnn_full_lopo_density")[tI]),
                          ("Final model",J("patchcnn_wide_lopo_density")[tS],J("patchcnn_wide_lopo_density")[tI])])]
fig,axes=plt.subplots(1,2,figsize=(14,4.4),gridspec_kw={"width_ratios":[5,2.2]},sharey=True)
for ax,(title,models) in zip(axes,panels):
    x=np.arange(len(models)); w=0.38
    for k,(lab,col) in enumerate([("trained on iPhone, tested on Samsung","C1"),("trained on Samsung, tested on iPhone","C3")]):
        vals=[m[1+k]["auc"] for m in models]; lo=[m[1+k]["auc"]-m[1+k]["auc_ci"][0] for m in models]; hi=[m[1+k]["auc_ci"][1]-m[1+k]["auc"] for m in models]
        ax.bar(x+(k-0.5)*w,vals,w,yerr=[lo,hi],capsize=3,color=col,label=lab)
        for xi,m in zip(x,models): ax.text(xi+(k-0.5)*w,0.02,f"{round(m[1+k]['apcer@bpcer5']*m[1+k]['n_attacks'])}/{m[1+k]['n_attacks']}\nmissed",ha="center",va="bottom",fontsize=7,color="white")
    ax.axhline(0.5,color="k",ls=":",lw=1); ax.set(xticks=x,xticklabels=[m[0] for m in models],ylim=(0,1.08),title=title)
axes[0].set_ylabel("AUC on the camera never seen in training"); axes[0].text(4.45,0.51,"coin flip",fontsize=8); axes[0].legend(fontsize=8,loc="upper left")
fig.suptitle("Unseen-camera test: train on one phone, test on the other. Bars: AUC with 95 % interval. Text: attacks missed at 5 % BPCER.",fontsize=10)
plt.tight_layout(); plt.savefig(OUT,dpi=120); plt.close(); print("wrote",OUT)
