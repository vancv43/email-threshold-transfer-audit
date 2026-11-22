"""Shared loading and metrics for PROTOCOL.md v0.2 with amendments A1-A3."""
import json, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
HEADER_FIELDS = ("subject", "from_name", "from_domain", "content_type")  # amendment A4


def load_messages():
    df = pd.read_parquet(P / "messages.parquet")
    df["year"] = df["year_month"].str[:4].astype(int)
    hdr = df["header"].map(lambda h: {k: v for k, v in json.loads(h).items() if k in HEADER_FIELDS})
    df["subject"] = hdr.map(lambda h: h.get("subject", ""))
    df["text"] = df["subject"] + "\n" + df["body"].str[:5000]
    return df, hdr


def split_frame(df, seed):
    sp = pd.read_parquet(P / "splits.parquet")
    sp = sp[sp["seed"] == seed]
    return {name: df.loc[df["msg_id"].isin(g["msg_id"])].reset_index(drop=True) for name, g in sp.groupby("split")}


def tpr_at_fpr(y, s, fpr_max):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_max
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(y_cal, s_cal, fpr_max):
    """Smallest score threshold whose FPR on the calibration ham does not exceed fpr_max."""
    ham = np.sort(s_cal[y_cal == 0])[::-1]
    k = int(np.floor(fpr_max * len(ham)))  # number of ham allowed above the threshold
    return float(ham[k]) if k < len(ham) else float(ham[-1])


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def metrics(y, s, p, thr):
    """y labels, s raw scores, p calibrated probabilities, thr threshold set on the calibration split."""
    y = np.asarray(y)
    out = {"n": int(len(y)), "n_spam": int(y.sum())}
    if 0 < y.sum() < len(y):
        out.update(tpr_fpr_0p1=tpr_at_fpr(y, s, 0.001), tpr_fpr_1=tpr_at_fpr(y, s, 0.01),
                   roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)))
    pred = (s > thr).astype(int)
    out.update(f1=float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)), ece=ece(y, p),
               cal_thr_tpr=float(pred[y == 1].mean()) if y.sum() else None,
               cal_thr_fpr=float(pred[y == 0].mean()) if (y == 0).any() else None)
    return out
