"""Accuracy, precision, recall, F1 and ROC-AUC of every finished model on the A7 data, at the operating point of
the protocol: the score threshold set on the calibration split for FPR 0.1%.

Classical baselines: counts rebuilt from n, n_spam and the TPR/FPR at that threshold stored per seed.
Transformer baselines: computed from the saved scores. Output: results/summary_operating_point_A7.json
"""
import glob, json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, threshold_at_fpr

SETS = ["s1", "s2", "s3"]


def from_rates(n, n_spam, tpr, fpr):
    n_ham = n - n_spam
    tp, fp = tpr * n_spam, fpr * n_ham
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"accuracy": (tp + n_ham - fp) / n, "precision": prec, "recall": tpr,
            "f1": 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0}


runs = {}  # model -> set -> list of per-seed dicts
for f in sorted(glob.glob(str(ROOT / "results/baselines/classical_seed*.json"))):
    for m, v in json.load(open(f))["models"].items():
        for s in SETS:
            t = v["test"][s]
            r = from_rates(t["n"], t["n_spam"], t["cal_thr_tpr"], t["cal_thr_fpr"]); r["auc"] = t["roc_auc"]
            runs.setdefault(m, {}).setdefault(s, []).append(r)
for f in sorted(glob.glob(str(ROOT / "results/gpu7/jnsm-tb7-*/out/scores_*_best.parquet"))):
    m = f.split("jnsm-tb7-")[1].split("/")[0].rsplit("-s", 1)[0]
    g = {k: v for k, v in pd.read_parquet(f).groupby("set")}
    thr = threshold_at_fpr(g["cal"]["label"].values, g["cal"]["score"].values, 0.001)
    for s in SETS:
        d = g[f"{s}_test"]; y, sc = d["label"].values, d["score"].values; p = sc > thr
        r = from_rates(len(y), int(y.sum()), p[y == 1].mean(), p[y == 0].mean()); r["auc"] = float(roc_auc_score(y, sc))
        runs.setdefault(m, {}).setdefault(s, []).append(r)
out = {m: {s: {"n_seeds": len(rs), **{k: {"mean": float(np.mean([r[k] for r in rs])),
                                           "sd": float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else None}
                                       for k in rs[0]}} for s, rs in v.items()} for m, v in runs.items()}
(ROOT / "results/summary_operating_point_A7.json").write_text(json.dumps(out, indent=1))
for s in SETS:
    print(f"\n{s.upper()}  model | seeds | acc | prec | rec | f1 | auc")
    for m, v in out.items():
        r = v[s]
        print(m, r["n_seeds"], *[f"{r[k]['mean']:.3f}" + (f"±{r[k]['sd']:.3f}" if r[k]["sd"] is not None else "")
                                 for k in ["accuracy", "precision", "recall", "f1", "auc"]], sep=" | ")
