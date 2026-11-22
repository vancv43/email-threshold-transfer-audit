"""Analyses added for the internal review of the JISA manuscript (saved scores only, no training).

- tab_tcal: TPR and FPR at the threshold fixed on calibration, per test set, plus standardized partial ROC-AUC on S2
  (FPR <= 1%) and mean confusion counts on S2 for MuST-Guard.                                         [M1, M2]
- tab_stats_h: MuST-Guard minus each model, TPR at FPR 0.1%, two-stage bootstrap that resamples training seeds
  and test messages (message identity kept across seeds and models), threshold re-estimated in every replicate;
  unadjusted 95% interval, Holm-adjusted p; seed-level paired differences, their SD and d_z.          [M5, m1, m2]
- tab_recal_corpus: recalibration on SpamAssassin ham and on Enron-Spam ham separately, with the order-statistic
  reference Beta(k+1, n-k) for the out-of-sample FPR of an empirical quantile threshold.              [M8]
- tab_conf_class: conformal coverage of MuST-Guard by class and quarantine load per 10,000 messages; risk-coverage
  curves in conf_rc.json.                                                                            [M9]
- tab_g2flips_seed: flips of G2 spam at the calibration threshold per seed, no pooling across seeds.  [m3]
- tab_year2 / tab_year_head2: S3 by year with the year of the split data, totals equal to Table 1.     [numbers]
- tab_flow: data flow from parsed messages to the splits of seed 0.                                   [R1, M11]
Output: ../manuscript/tables/*.tex and ../results_revision.json
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parents[1]
STUDY = HERE
sys.path.insert(0, str(STUDY / "code"))
exec((STUDY / "code/a11/analyze_a11.py").read_text().split("# per-run point metrics")[0])  # runs, split, per_set, tpr_fast
from common import threshold_at_fpr

OUT = HERE / "results/tables"
ORDER = ["Multinomial naive Bayes", "Logistic regression", "Linear SVM", "BERT-base, earlier pipeline", "DistilBERT",
         "RoBERTa-base", "DeBERTa-v3-base", "MuST-Guard v2", "MuST-Guard v1"]
NAME = {"MuST-Guard v1": "MuST-Guard (ours)", "MuST-Guard v2": "MuST-Guard-SA (ours)", "BERT-base, earlier pipeline": "BERT-base"}
nm = lambda n: NAME.get(n, n)
SETS = ["s1_test", "s3_test", "g2_rewrite", "s2_test"]
P = "MuST-Guard v1"
rng = np.random.default_rng(20261119)
res = {}


def ms(v, d=1, f=100):
    v = f * np.asarray(v, float)
    return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def tcal(df):
    c = df[df["set"] == "cal"]
    return threshold_at_fpr(c["label"].values, c["score"].values, 0.001)


# M1, M2: operating point at t_cal
rows, conf = [], {}
for n in ORDER:
    cells = []
    for s in SETS:
        t_, f_ = [], []
        for sd, df in sorted(runs[n].items()):
            h, p = split(per_set(df, s)); t = tcal(df)
            t_.append((p > t).mean()); f_.append((h > t).mean())
        cells += [ms(t_), ms(f_)]
        res.setdefault("tcal", {})[f"{n}|{s}"] = [float(np.mean(t_)), float(np.mean(f_))]
    pa = []
    for sd, df in sorted(runs[n].items()):
        d = per_set(df, "s2_test"); pa.append(roc_auc_score(d["label"].values, d["score"].values, max_fpr=0.01))
    res.setdefault("pauc_s2", {})[n] = float(np.mean(pa))
    rows.append(f"{nm(n)} & " + " & ".join(cells) + f" & {100 * np.mean(pa):.1f} \\\\")
    if n == P:
        cm = []
        for sd, df in sorted(runs[n].items()):
            h, p = split(per_set(df, "s2_test")); t = tcal(df)
            cm.append([(p > t).sum(), (p <= t).sum(), (h > t).sum(), (h <= t).sum()])
        conf = dict(zip(["TP", "FN", "FP", "TN"], np.mean(cm, axis=0).round().astype(int).tolist()))
(OUT / "tab_tcal.tex").write_text("\n".join(rows) + "\n"); res["confusion_s2_mg"] = conf

# M5: two-stage bootstrap over seeds and messages
B = 1000
rows = []
for s in SETS:
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{ {'s1_test': 'S1 in-distribution', 's3_test': 'S3 temporal', 'g2_rewrite': 'G2 rewrites', 's2_test': 'S2 cross-corpus'}[s] }}}}} \\\\")
    comp = []
    for n in ORDER:
        if n == P:
            continue
        common = sorted(set(runs[n]) & set(runs[P]))
        A = {sd: split(per_set(runs[P][sd], s)) for sd in common}
        ref = {sd: per_set(runs[P][sd], s).index for sd in common}
        Bm = {sd: split(per_set(runs[n][sd], s).loc[ref[sd]]) for sd in common}
        point = np.array([tpr_fast(*A[sd]) - tpr_fast(*Bm[sd]) for sd in common])
        draws = np.empty(B)
        shared = s != "s1_test"
        for b in range(B):
            sds = rng.choice(common, len(common), replace=True)
            if shared:
                nh, ns = len(A[common[0]][0]), len(A[common[0]][1])
                ih, ip = rng.integers(0, nh, nh), rng.integers(0, ns, ns)
            vals = []
            for sd in sds:
                if not shared:
                    nh, ns = len(A[sd][0]), len(A[sd][1]); ih, ip = rng.integers(0, nh, nh), rng.integers(0, ns, ns)
                vals.append(tpr_fast(A[sd][0][ih], A[sd][1][ip]) - tpr_fast(Bm[sd][0][ih], Bm[sd][1][ip]))
            draws[b] = np.mean(vals)
        p = float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()) + 1 / B))
        sdv = point.std(ddof=1) if len(point) > 1 else float("nan")
        dz = point.mean() / sdv if len(point) == 5 and sdv > 0 else float("nan")
        comp.append(dict(n=n, k=len(common), diff=point.mean(), lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5),
                         p=p, sd=sdv, dz=dz, seeds=point.tolist()))
    o = np.argsort([c["p"] for c in comp]); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(comp) - r) * comp[i]["p"])); comp[i]["ph"] = run
    for c in comp:
        pf = "$<$0.001" if c["ph"] < 0.001 else f"{c['ph']:.3f}"
        sd_ = "--" if not np.isfinite(c["sd"]) else f"{100 * c['sd']:.2f}"
        dz_ = "--" if not np.isfinite(c["dz"]) else f"{c['dz']:+.2f}"
        rows.append(f"\\quad {nm(c['n'])} & {c['k']} & {100 * c['diff']:+.2f} & [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}] & {pf} & {sd_} & {dz_} \\\\")
        res.setdefault("stats_h", {})[f"{c['n']}|{s}"] = {k: (v if not isinstance(v, (np.floating, float)) else float(v)) for k, v in c.items()}
(OUT / "tab_stats_h.tex").write_text("\n".join(rows) + "\n")

# M8: recalibration per target corpus with the order-statistic reference
meta = pd.read_parquet(STUDY / "data/processed_A7/messages.parquet", columns=["msg_id", "source"]).set_index("msg_id")["source"]
plan = [("SpamAssassin", "spamassassin", [1000, 2000]), ("Enron-Spam", None, [1000, 5000, 10000])]
R = 50; rows = []
ref_cells = []
for corp, src, NS in plan:
    for k_ in NS:
        j = int(np.floor(0.001 * k_))
        ref_cells.append(f"{100 * (j + 1) / (k_ + 1):.2f} / {100 * beta.ppf(0.95, j + 1, k_ - j):.2f} / --")
for n in ORDER:
    cells = []
    for corp, src, NS in plan:
        for k_ in NS:
            f_, t_ = [], []
            for sd, df in sorted(runs[n].items()):
                d = per_set(df, "s2_test"); so = meta.reindex(d.index).values
                h = d["score"].values[(d["label"].values == 0) & ((so == src) if src else pd.isna(so))]
                pp = d["score"].values[d["label"].values == 1]
                for _ in range(R):
                    idx = rng.permutation(len(h)); c, e = h[idx[:k_]], h[idx[k_:]]
                    thr = np.sort(c)[::-1][int(np.floor(0.001 * k_))]
                    f_.append((e > thr).mean()); t_.append((pp > thr).mean())
            cells.append(f"{100 * np.mean(f_):.2f} / {100 * np.percentile(f_, 95):.2f} / {100 * np.mean(t_):.1f}")
            res.setdefault("recal_corpus", {})[f"{n}|{corp}|{k_}"] = [float(np.mean(f_)), float(np.percentile(f_, 95)), float(np.mean(t_))]
    rows.append(f"{nm(n)} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule\nOrder-statistic reference & " + " & ".join(ref_cells) + " \\\\")
(OUT / "tab_recal_corpus.tex").write_text("\n".join(rows) + "\n")

# M9: class-conditional conformal coverage and quarantine load
a9 = json.load(open(STUDY / "results/mustguard/summary_confirm_a9.json"))
T = {i: r["temperature"] for i, r in enumerate(a9["per_run"])}
rows, rc = [], {}
for s, lab in [("s1_test", "S1"), ("s3_test", "S3"), ("g2_rewrite", "G2"), ("s2_test", "S2")]:
    for al in (0.05, 0.01):
        cov_h, cov_s, empty, load = [], [], [], []
        for sd, df in sorted(runs[P].items()):
            Tm = T[sd]; c = df[df["set"] == "cal"]
            r_cal = -(2 * c["label"].values - 1) * c["score"].values.astype(np.float64) / Tm
            nc = len(r_cal); q = np.sort(r_cal)[int(np.ceil((nc + 1) * (1 - al))) - 1]
            d = per_set(df, s); y = d["label"].values; m_ = d["score"].values.astype(np.float64) / Tm
            in1, in0 = (-m_ <= q), (m_ <= q)          # label 1 enters when -(+1)s/T <= q; label 0 when s/T <= q
            cov = np.where(y == 1, in1, in0)
            cov_h.append(cov[y == 0].mean()); cov_s.append(cov[y == 1].mean())
            emp = ~in1 & ~in0; empty.append(emp.mean()); load.append(10000 * emp.mean())
        rows.append(f"{lab} & {1 - al:.2f} & {ms(cov_h)} & {ms(cov_s)} & {ms(empty)} & {np.mean(load):,.0f} \\\\")
    curves = []
    for sd, df in sorted(runs[P].items()):
        d = per_set(df, s); y = d["label"].values; sc = d["score"].values; t = tcal(df)
        mg = np.abs(sc / T[sd]); o = np.argsort(-mg); err = ((sc > t).astype(int) != y)[o]
        k = np.arange(1, len(o) + 1); grid = np.linspace(0.05, 1, 20)
        curves.append([float(err[: max(1, int(g * len(o)))].mean()) for g in grid])
    rc[lab] = {"coverage": np.linspace(0.05, 1, 20).tolist(), "risk": np.mean(curves, axis=0).tolist()}
(OUT / "tab_conf_class.tex").write_text("\n".join(rows) + "\n"); res["risk_coverage"] = rc

# m3: G2 flips per seed at the calibration threshold
rows = []
for n in ["BERT-base, earlier pipeline", "DistilBERT", "RoBERTa-base", "DeBERTa-v3-base", "MuST-Guard v2", "MuST-Guard v1"]:
    per = []
    for sd, df in sorted(runs[n].items()):
        t = tcal(df); o = per_set(df, "g2_original"); r = per_set(df, "g2_rewrite").loc[o.index]
        os_, rs_ = o["score"].values[o["label"].values == 1] > t, r["score"].values[r["label"].values == 1] > t
        per.append((int((os_ & ~rs_).sum()), int((~os_ & rs_).sum())))
    lost = "/".join(str(a) for a, _ in per); gained = "/".join(str(b) for _, b in per)
    rows.append(f"{nm(n)} & {len(per)} & {lost} & {gained} & {sum(b > a for a, b in per)} of {len(per)} \\\\")
(OUT / "tab_g2flips_seed.tex").write_text("\n".join(rows) + "\n")

# numbers: S3 by year on the split data, and the data flow of seed 0
flow = json.load(open(HERE / "code/flow_seed0.json"))
yr = flow["s3_by_year"]
years = sorted(yr["0"])
rows = []
for n in ORDER:
    cells = []
    for y_ in years:
        v = []
        for sd, df in sorted(runs[n].items()):
            d = per_set(df, "s3_test"); keep = d.index.isin(flow["s3_ids_by_year"][y_])
            v.append(tpr_fast(d["score"].values[keep & (d["label"].values == 0)], d["score"].values[keep & (d["label"].values == 1)]))
        cells.append(f"{100 * np.mean(v):.1f}")
    rows.append(f"{nm(n)} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule\nHam & " + " & ".join(f"{yr['0'][y_]:,}" for y_ in years) + " \\\\")
rows.append("Spam & " + " & ".join(f"{yr['1'][y_]:,}" for y_ in years) + " \\\\")
(OUT / "tab_year2.tex").write_text("\n".join(rows) + "\n"); (OUT / "tab_year_head2.tex").write_text(" & ".join(years) + "\n")
(HERE / "results_revision.json").write_text(json.dumps(res, indent=1, default=float))
print("confusion S2 MuST-Guard per seed mean:", conf)
print("pAUC S2:", {nm(k): round(100 * v, 1) for k, v in res["pauc_s2"].items()})
