"""Tables of the JISA manuscript that are not produced by code/report/manuscript_tables.py of the study folder.

Reads only saved per-message scores (study folder, PROTOCOL.md A10-A12); no training.
- tab_tpr: TPR at FPR 0.1% on S1, S3, G2 and S2, realized S2 FPR at the calibration threshold, seed count.
- tab_std_a / tab_std_b: accuracy, precision, recall, F1-score and ROC-AUC at the default decision threshold.
- tab_dz: Cohen's d_z of the per-seed TPR difference, MuST-Guard minus each baseline, on the seeds both finished.
- tab_recal: target-domain recalibration, threshold set on n cross-corpus ham drawn at random, scored on the rest.
Rows: seven baselines, MuST-Guard-SA, MuST-Guard (full) last. Output: ../manuscript/tables/*.tex, ../results_jisa.json
"""
import json, pathlib, re, sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

STUDY = pathlib.Path(__file__).resolve().parents[3] / ".." / "Springer/Journal of Network and Systems Management"
STUDY = STUDY.resolve()
sys.path.insert(0, str(STUDY / "code"))
exec((STUDY / "code/a11/analyze_a11.py").read_text().split("# per-run point metrics")[0])  # loader: runs, split, per_set, tpr_fast
from common import threshold_at_fpr

OUT = pathlib.Path(__file__).resolve().parents[1] / "manuscript/tables"; OUT.mkdir(parents=True, exist_ok=True)
NAME = {"MuST-Guard v1": "MuST-Guard", "MuST-Guard v2": "MuST-Guard-SA"}
ORDER = ["Multinomial naive Bayes", "Logistic regression", "Linear SVM", "BERT-base, earlier pipeline", "DistilBERT",
         "RoBERTa-base", "DeBERTa-v3-base", "MuST-Guard v2", "MuST-Guard v1"]
LABEL = {"BERT-base, earlier pipeline": "BERT-base"}
SETS = ["s1_test", "s3_test", "g2_rewrite", "s2_test"]
show = lambda n: NAME.get(n, LABEL.get(n, n)) + (" (ours)" if n in NAME else "")
res = {}


def ms(v, d=1):
    v = 100 * np.asarray(v)
    return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def cal_thr(df):
    c = df[df["set"] == "cal"]
    return threshold_at_fpr(c["label"].values, c["score"].values, 0.001)


# TPR at FPR 0.1% and realized cross-corpus FPR
rows = []
for n in ORDER:
    rs = runs[n]; cells = []
    for s in SETS:
        cells.append(ms([tpr_fast(*split(per_set(rs[sd], s))) for sd in sorted(rs)]))
    fpr = [float((split(per_set(rs[sd], "s2_test"))[0] > cal_thr(rs[sd])).mean()) for sd in sorted(rs)]
    res.setdefault("tpr", {})[n] = cells + [ms(fpr)]
    rows.append(f"{show(n)} & {len(rs)} & " + " & ".join(cells) + f" & {ms(fpr)} \\\\")
(OUT / "tab_tpr.tex").write_text("\n".join(rows) + "\n")

# standard metrics at the default threshold, score > 0
std = {}
for n in ORDER:
    for s in SETS:
        M = []
        for sd, df in sorted(runs[n].items()):
            d = per_set(df, s); y = d["label"].values; sc = d["score"].values; p = (sc > 0).astype(int)
            M.append([accuracy_score(y, p), precision_score(y, p, zero_division=0), recall_score(y, p), f1_score(y, p),
                      roc_auc_score(y, sc)])
        std[(n, s)] = np.array(M)
for tab, pair in (("tab_std_a", ["s1_test", "s3_test"]), ("tab_std_b", ["g2_rewrite", "s2_test"])):
    rows = []
    for n in ORDER:
        cells = [f"{100 * std[(n, s)][:, j].mean():.2f}" for s in pair for j in range(5)]
        rows.append(f"{show(n)} & " + " & ".join(cells) + " \\\\")
    (OUT / f"{tab}.tex").write_text("\n".join(rows) + "\n")
res["std"] = {f"{n}|{s}": v.tolist() for (n, s), v in std.items()}

# Cohen's d_z of per-seed TPR differences, proposed minus baseline
P = "MuST-Guard v1"; rows = []
for n in ORDER[:-1]:
    cells = []
    for s in SETS:
        common = sorted(set(runs[n]) & set(runs[P]))
        diff = np.array([tpr_fast(*split(per_set(runs[P][sd], s))) - tpr_fast(*split(per_set(runs[n][sd], s))) for sd in common])
        dz = diff.mean() / diff.std(ddof=1) if len(diff) == 5 and diff.std(ddof=1) > 0 else float("nan")  # d_z only on five shared seeds
        cells.append(f"{100 * diff.mean():+.1f} / {dz:+.2f}" if np.isfinite(dz) else f"{100 * diff.mean():+.1f} / --")
        res.setdefault("dz", {})[f"{n}|{s}"] = [float(diff.mean()), float(dz), common]
    rows.append(f"{show(n)} & {len(set(runs[n]) & set(runs[P]))} & " + " & ".join(cells) + " \\\\")
(OUT / "tab_dz.tex").write_text("\n".join(rows) + "\n")

# target-domain recalibration on the cross-corpus test
NS, R = [200, 1000, 2000, 5000, 10000], 50
rng = np.random.default_rng(20261116); rec = {}
for n in ORDER:
    for k in NS:
        f_, t_ = [], []
        for sd, df in sorted(runs[n].items()):
            h, p = split(per_set(df, "s2_test"))
            for _ in range(R):
                idx = rng.permutation(len(h)); c, e = h[idx[:k]], h[idx[k:]]
                j = int(np.floor(0.001 * k)); thr = np.sort(c)[::-1][j]
                f_.append(float((e > thr).mean())); t_.append(float((p > thr).mean()))
        rec[(n, k)] = (np.mean(f_), np.percentile(f_, 95), np.mean(t_))
rows = []
for n in ORDER:
    rows.append(f"{show(n)} & " + " & ".join(f"{100 * rec[(n, k)][0]:.2f} / {100 * rec[(n, k)][1]:.2f} / {100 * rec[(n, k)][2]:.1f}"
                                         for k in [1000, 5000, 10000]) + " \\\\")
(OUT / "tab_recal.tex").write_text("\n".join(rows) + "\n")
res["recal"] = {f"{n}|{k}": list(v) for (n, k), v in rec.items()}
res["seeds"] = {n: sorted(runs[n]) for n in ORDER}
(pathlib.Path(__file__).resolve().parents[1] / "results_jisa.json").write_text(json.dumps(res, indent=1))
for n in ORDER:
    print(f"{show(n):28s}", " | ".join(f"n={k}: FPR {100*rec[(n,k)][0]:.2f}% p95 {100*rec[(n,k)][1]:.2f}% TPR {100*rec[(n,k)][2]:.1f}%" for k in NS))

# merged tables for layout: study tab_stats with d_z appended, study tab_g2 with the flips of tab_flips
ST = STUDY / "manuscript/tables"
SCN = {"In-distribution": "s1_test", "Temporal": "s3_test", "Cross-corpus": "s2_test", "$G_2$ rewrites": "g2_rewrite"}
rows, scen = [], None
for l in (ST / "tab_stats.tex").read_text().strip().split("\n"):
    if l.startswith("\\multicolumn"):
        scen = SCN[re.search(r"\\emph\{([^}]*)\}", l).group(1)]
        rows.append(l.replace("{6}", "{7}")); continue
    name = l.split("&")[0].replace("\\quad", "").strip()
    d = res["dz"].get(f"{name}|{scen}")
    dz = f"{d[1]:+.2f}" if d and np.isfinite(d[1]) else "--"
    rows.append(l.rstrip().rstrip("\\").rstrip() + f" & {dz} \\\\")
(OUT / "tab_stats_dz.tex").write_text("\n".join(rows) + "\n")
fl = {l.split("&")[0].strip(): [c.strip().rstrip("\\").strip() for c in l.split("&")[2:]] for l in (ST / "tab_flips.tex").read_text().strip().split("\n")}
rows = []
for l in (ST / "tab_g2.tex").read_text().strip().split("\n"):
    name = l.split("&")[0].strip(); f = fl.get(name, ["--", "--", "--"])
    rows.append(l.rstrip().rstrip("\\").rstrip() + " & " + " & ".join(f) + " \\\\")
(OUT / "tab_g2flips.tex").write_text("\n".join(rows) + "\n")
