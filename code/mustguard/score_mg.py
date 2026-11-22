"""Metrics of the MuST-Guard runs from their saved scores (PROTOCOL.md section 4, A3, A5, A8).

Per run: TPR@FPR 0.1% and 1%, PR-AUC, ROC-AUC, and at the calibration threshold for FPR 0.1% the TPR, FPR,
accuracy, precision, recall and F1, on S1, S2, S3 and the G2 sets.
Calibrated abstention: a temperature is fitted on val (NLL), then split conformal on cal; for alpha in
{0.01, 0.05} a message is accepted when its prediction set has one label, and the report gives coverage,
abstention rate and the TPR/FPR of the accepted messages. Nonconformity: the signed log-odds margin of the true
class (A9, primary, keys conformal_logit_<alpha>) and 1 - p(true class) of A8 (kept, keys conformal_<alpha>),
which ties at 0 once the probabilities round to 0 or 1. Gate weights are averaged
per set and class. Output: results/mustguard/summary_<name>.json
Usage: python score_mg.py <runs_glob> <name>
"""
import glob, json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, threshold_at_fpr, tpr_at_fpr

TESTS = ["s1_test", "s2_test", "s3_test", "g2_original", "g2_rewrite"]
GATES = ["gate_body", "gate_header", "gate_url", "gate_struct"]


def sig(x):
    return 1 / (1 + np.exp(-x))


def fit_temperature(y, s):
    def nll(t):
        p = np.clip(sig(s / t), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 100), method="bounded").x)


def quantile(nc, alpha):
    n = len(nc)
    return np.quantile(nc, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def conformal(y_cal, p_cal, y, p, alpha):
    q = quantile(1 - np.where(y_cal == 1, p_cal, 1 - p_cal), alpha)
    return sets(y, (1 - p) <= q, p <= q, q)  # label 1 in the set, label 0 in the set


def conformal_logit(y_cal, z_cal, y, z, alpha):
    q = quantile(-(2 * y_cal - 1) * z_cal, alpha)  # A9: nonconformity -(2y-1) z, z = s / T
    return sets(y, -z <= q, z <= q, q)


def sets(y, in1, in0, q):
    single = in1 ^ in0
    covered = np.where(y == 1, in1, in0)
    acc = single & in1
    return {"q": float(q), "coverage": float(covered.mean()), "abstain": float(1 - single.mean()),
            "accepted_tpr": float(acc[(y == 1) & single].mean()) if ((y == 1) & single).any() else None,
            "accepted_fpr": float(acc[(y == 0) & single].mean()) if ((y == 0) & single).any() else None}


def at_threshold(y, s, thr):
    pred = s > thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / max(1, int((y == 1).sum()))
    return {"cal_thr_tpr": rec, "cal_thr_fpr": fp / max(1, int((y == 0).sum())), "accuracy": float((pred == y).mean()),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def main():
    runs, name = sorted(glob.glob(sys.argv[1])), sys.argv[2]
    assert runs, "no score files"
    res = {"runs": runs, "per_run": []}
    for f in runs:
        g = {k: v for k, v in pd.read_parquet(f).groupby("set")}
        y_cal, s_cal = g["cal"]["label"].values, g["cal"]["score"].values
        thr = threshold_at_fpr(y_cal, s_cal, 0.001)
        T = fit_temperature(g["val"]["label"].values, g["val"]["score"].values)
        out = {"file": f, "temperature": T}
        for k in TESTS:
            y, s = g[k]["label"].values, g[k]["score"].values
            r = {"n": int(len(y)), "n_spam": int(y.sum()), "tpr_fpr_0p1": tpr_at_fpr(y, s, 0.001),
                 "tpr_fpr_1": tpr_at_fpr(y, s, 0.01), "pr_auc": float(average_precision_score(y, s)),
                 "roc_auc": float(roc_auc_score(y, s)), **at_threshold(y, s, thr),
                 "gates": {c: g[k].groupby("label")[c].mean().round(4).to_dict() for c in GATES}}
            for al in (0.01, 0.05):
                r[f"conformal_{al}"] = conformal(y_cal, sig(s_cal / T), y, sig(s / T), al)
                r[f"conformal_logit_{al}"] = conformal_logit(y_cal, s_cal / T, y, s / T, al)
            out[k] = r
        out["g2_drop_tpr_fpr_0p1"] = out["g2_original"]["tpr_fpr_0p1"] - out["g2_rewrite"]["tpr_fpr_0p1"]
        res["per_run"].append(out)
    keys = ["tpr_fpr_0p1", "tpr_fpr_1", "pr_auc", "roc_auc", "cal_thr_tpr", "cal_thr_fpr", "accuracy", "precision",
            "recall", "f1"]
    res["mean_sd"] = {k: {m: {"mean": float(np.mean([r[k][m] for r in res["per_run"]])),
                              "sd": float(np.std([r[k][m] for r in res["per_run"]], ddof=1)) if len(runs) > 1 else None}
                          for m in keys} for k in TESTS}
    res["conformal_mean_sd"] = {k: {f"{c}_{al}": {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                                                  for m in ("coverage", "abstain", "accepted_tpr", "accepted_fpr")
                                                  for v in [[r[k][f"{c}_{al}"][m] for r in res["per_run"] if r[k][f"{c}_{al}"][m] is not None]] if v}
                                    for c in ("conformal", "conformal_logit") for al in (0.01, 0.05)} for k in TESTS}
    o = ROOT / "results/mustguard"; o.mkdir(parents=True, exist_ok=True)
    (o / f"summary_{name}.json").write_text(json.dumps(res, indent=1))
    for k in TESTS:
        print(k, {m: round(v["mean"], 4) for m, v in res["mean_sd"][k].items()})


if __name__ == "__main__":
    main()
