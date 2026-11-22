"""Analyses of PROTOCOL.md A10 that need only the saved per-message scores (no training).

Runs with saved scores: the transformer baselines (results/gpu7/) and the proposed model (results/mustguard/confirm/).
The classical baselines saved aggregates only; their per-year and per-source numbers come from their JSON files, and
their per-message scores are produced by the A10 batch jobs.
- Stratified per-message bootstrap, 10,000 resamples, of the seed-mean TPR at FPR 0.1%; the S2, S3 and G2 sets are the
  same for every seed, so one resample is shared by all seeds and models (paired); S1 differs per seed and is resampled
  per seed. Proposed minus baseline on the seeds both have, two-sided bootstrap p, Holm across baselines per scenario.
- McNemar at the calibration threshold: discordant counts summed over the common seeds, exact binomial test.
- TPR at FPR 1%, PR-AUC; S3 per test year; S2 per corpus; positives split into spam and phishing; G2 flips;
  conformal set composition (A9 nonconformity); number of ham behind each 0.1% threshold; per-seed realized FPR.
Output: results/extra_analysis.json and manuscript/tables/tab_*.tex
Usage: python extra_analysis.py
"""
import glob, json, re, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "mustguard"))
from common import ROOT, threshold_at_fpr, tpr_at_fpr
from score_mg import fit_temperature, quantile

B, RNG = 10000, np.random.default_rng(20261111)
OUT = ROOT / "manuscript/tables"
SETS = {"s1_test": "In-distribution", "s2_test": "Cross-corpus", "s3_test": "Temporal", "g2_rewrite": "$G_2$ rewrites"}
PROPOSED = "Proposed"
CLASSICAL = {"MultinomialNB": "Multinomial naive Bayes", "LR": "Logistic regression", "LinearSVM": "Linear SVM"}
MODELS = {"bert-base-cased": "BERT-base, earlier pipeline", "distilbert-base-uncased": "DistilBERT",
          "roberta-base": "RoBERTa-base", "deberta-v3-base": "DeBERTa-v3-base"}
res = {}


def load():
    runs = {}
    for key, name in MODELS.items():
        for f in glob.glob(str(ROOT / f"results/gpu7/jnsm-tb7-{key}-s*/out/scores_*_best.parquet")):
            runs.setdefault(name, {})[int(re.search(r"_s(\d)_best", f).group(1))] = pd.read_parquet(f)
    for f in glob.glob(str(ROOT / "results/mustguard/confirm/s*/out/scores_*_best.parquet")):
        runs.setdefault(PROPOSED, {})[int(re.search(r"_s(\d)_best", f).group(1))] = pd.read_parquet(f)
    for f in glob.glob(str(ROOT / "results/a10/classical_scores_seed*.parquet")):  # A10 batch jobs, when collected
        d = pd.read_parquet(f)
        for key, name in CLASSICAL.items():
            runs.setdefault(name, {})[int(d["seed"].iloc[0])] = d[d["model"] == key].drop(columns=["model", "seed"])
    return runs


runs = load()
meta = pd.read_parquet(ROOT / "data/processed_A7/messages.parquet", columns=["msg_id", "source", "year_month"])
meta["year"] = meta["year_month"].str[:4].astype(int)
SRC, YEAR = meta.set_index("msg_id")["source"], meta.set_index("msg_id")["year"]


def tpr_fast(ham, spam, fpr=0.001):
    """TPR at the largest threshold whose FPR does not exceed fpr; equals common.tpr_at_fpr without ties."""
    k = int(np.floor(fpr * len(ham)))
    t = np.partition(ham, len(ham) - 1 - k)[len(ham) - 1 - k]
    return float((spam > t).mean())


def per_set(df, s):
    d = df[df["set"] == s]
    return d.set_index("msg_id")


# check the fast estimator against the reference implementation on every run and set
for name, rs in runs.items():
    for sd, df in rs.items():
        for s in SETS:
            d = per_set(df, s); y, sc = d["label"].values, d["score"].values
            ham_, spam_ = sc[y == 0], sc[y == 1]
            k_ = int(np.floor(0.001 * len(ham_))); t_ = np.partition(ham_, len(ham_) - 1 - k_)[len(ham_) - 1 - k_]
            tie = (spam_ == t_).any()  # half-precision scores can tie at the threshold; the rule flags s > t
            assert tie or abs(tpr_fast(ham_, spam_) - tpr_at_fpr(y, sc, 0.001)) < 1e-9, (name, sd, s)
            assert (ham_ > t_).sum() <= k_, (name, sd, s)

# 1. bootstrap CIs, paired differences, Holm
boot = {}
for s in SETS:
    seeds_all = sorted({sd for rs in runs.values() for sd in rs})
    ref = {sd: per_set(runs[PROPOSED][sd], s) for sd in runs[PROPOSED]}
    common_set = s != "s1_test"
    draws = {name: {sd: np.empty(B) for sd in rs} for name, rs in runs.items()}
    arr = {}
    for name, rs in runs.items():
        for sd, df in rs.items():
            d = per_set(df, s).loc[ref[sd].index]  # same message order as the proposed run of that seed
            arr[(name, sd)] = (d["score"].values[d["label"].values == 0], d["score"].values[d["label"].values == 1])
    for b0 in range(0, B, 500):
        nb = min(500, B - b0)
        idx = {}
        for sd in seeds_all:
            if sd not in ref:
                continue
            y = ref[sd]["label"].values
            nh, ns = int((y == 0).sum()), int((y == 1).sum())
            if common_set and idx:
                idx[sd] = next(iter(idx.values()))
            else:
                idx[sd] = (RNG.integers(0, nh, (nb, nh)), RNG.integers(0, ns, (nb, ns)))
        for (name, sd), (h, p) in arr.items():
            ih, ip = idx[sd]
            for j in range(nb):
                draws[name][sd][b0 + j] = tpr_fast(h[ih[j]], p[ip[j]])
    boot[s] = draws

def holm_adjust(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj.tolist()


cis, tests = {}, {}
for s, draws in boot.items():
    cis[s] = {}
    for name, d in draws.items():
        point = np.mean([tpr_fast(*arr_) for arr_ in [(lambda x: (x["score"].values[x["label"].values == 0],
                                                                    x["score"].values[x["label"].values == 1]))(per_set(runs[name][sd], s)) for sd in d]])
        m = np.mean(np.stack(list(d.values())), axis=0)
        cis[s][name] = {"seeds": sorted(d), "mean": float(point), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}
    rows = []
    for name in draws:
        if name == PROPOSED:
            continue
        common = sorted(set(draws[name]) & set(draws[PROPOSED]))
        diff = np.mean([draws[PROPOSED][sd] - draws[name][sd] for sd in common], axis=0)
        pt = np.mean([tpr_fast(*(lambda x: (x["score"].values[x["label"].values == 0], x["score"].values[x["label"].values == 1]))(per_set(runs[PROPOSED][sd], s)))
                      - tpr_fast(*(lambda x: (x["score"].values[x["label"].values == 0], x["score"].values[x["label"].values == 1]))(per_set(runs[name][sd], s)))
                      for sd in common])
        p = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / B))
        # McNemar on decisions at each run's calibration threshold, discordant pairs summed over common seeds
        b_, c_ = 0, 0
        for sd in common:
            dec = {}
            for nm in (PROPOSED, name):
                df = runs[nm][sd]; cal = df[df["set"] == "cal"]
                thr = threshold_at_fpr(cal["label"].values, cal["score"].values, 0.001)
                d = per_set(df, s).loc[per_set(runs[PROPOSED][sd], s).index]
                dec[nm] = ((d["score"].values > thr).astype(int) == d["label"].values)
            b_ += int((dec[PROPOSED] & ~dec[name]).sum()); c_ += int((~dec[PROPOSED] & dec[name]).sum())
        pm = float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0
        rows.append({"baseline": name, "seeds": common, "diff": float(pt), "lo": float(np.percentile(diff, 2.5)),
                     "hi": float(np.percentile(diff, 97.5)), "p_boot": p, "mcnemar_b": b_, "mcnemar_c": c_, "p_mcnemar": pm})
    for key in ("p_boot", "p_mcnemar"):
        adj = holm_adjust([r[key] for r in rows])
        for r, a in zip(rows, adj):
            r[key + "_holm"] = a
    tests[s] = rows
res["bootstrap_ci"], res["paired_tests"] = cis, tests

# 2. secondary metrics, resolution, realized FPR per seed
sec, resol, rfpr = {}, {}, {}
for name, rs in runs.items():
    for s in SETS:
        v1, pr, fp = [], [], []
        for sd, df in rs.items():
            d = per_set(df, s); y, sc = d["label"].values, d["score"].values
            cal = df[df["set"] == "cal"]; thr = threshold_at_fpr(cal["label"].values, cal["score"].values, 0.001)
            v1.append(tpr_at_fpr(y, sc, 0.01)); pr.append(float(average_precision_score(y, sc)))
            fp.append(float((sc[y == 0] > thr).mean()))
            resol[s] = {"n_ham": int((y == 0).sum()), "n_spam": int((y == 1).sum()), "ham_allowed_at_0p1": int(np.floor(0.001 * (y == 0).sum()))}
        sec.setdefault(name, {})[s] = {"tpr_fpr_1": v1, "pr_auc": pr}
        rfpr.setdefault(name, {})[s] = fp
res["secondary"], res["resolution"], res["realized_fpr_per_seed"] = sec, resol, rfpr

# 3. S3 per year, S2 per corpus, spam vs phishing, from scores
def subset_tpr(df, s, mask_fn):
    d = per_set(df, s); m = mask_fn(d)
    y, sc = d["label"].values[m], d["score"].values[m]
    return tpr_at_fpr(y, sc, 0.001) if 0 < y.sum() < len(y) else None


years = sorted(YEAR.reindex(per_set(runs[PROPOSED][0], "s3_test").index).dropna().astype(int).unique())
per_year, per_corpus, per_class = {}, {}, {}
for name, rs in runs.items():
    for yv in years:
        per_year.setdefault(name, {})[int(yv)] = [subset_tpr(df, "s3_test", lambda d: (YEAR.reindex(d.index) == yv).values) for df in rs.values()]
    for corpus, fn in (("spamassassin", lambda d: (SRC.reindex(d.index) == "spamassassin").values),
                       ("enron_spam", lambda d: SRC.reindex(d.index).isna().values)):
        t, f = [], []
        for df in rs.values():
            cal = df[df["set"] == "cal"]; thr = threshold_at_fpr(cal["label"].values, cal["score"].values, 0.001)
            d = per_set(df, "s2_test"); m = fn(d); y, sc = d["label"].values[m], d["score"].values[m]
            t.append(tpr_at_fpr(y, sc, 0.001)); f.append(float((sc[y == 0] > thr).mean()))
        per_corpus.setdefault(name, {})[corpus] = {"tpr": t, "realized_fpr": f}
    for s in ("s3_test", "g2_rewrite", "g2_original"):
        for cls, src in (("spam", "untroubled"), ("phishing", "nazario")):
            per_class.setdefault(name, {}).setdefault(s, {})[cls] = [
                subset_tpr(df, s, lambda d, src=src: ((d["label"] == 0) | (SRC.reindex(d.index) == src)).values) for df in rs.values()]
res["s3_per_year"], res["s2_per_corpus"], res["per_class"] = per_year, per_corpus, per_class
res["s3_year_counts"] = per_set(runs[PROPOSED][0], "s3_test").assign(year=lambda d: YEAR.reindex(d.index).values).groupby(["year", "label"]).size().unstack().fillna(0).astype(int).to_dict()

# 4. G2 flips at the calibration threshold
flips = {}
for name, rs in runs.items():
    for sd, df in rs.items():
        cal = df[df["set"] == "cal"]; thr = threshold_at_fpr(cal["label"].values, cal["score"].values, 0.001)
        o = per_set(df, "g2_original"); w = per_set(df, "g2_rewrite").loc[o.index]
        sp = o["label"].values == 1
        do, dw = o["score"].values[sp] > thr, w["score"].values[sp] > thr
        lost, gained = int((do & ~dw).sum()), int((~do & dw).sum())
        flips.setdefault(name, []).append({"seed": sd, "lost": lost, "gained": gained,
                                           "p": float(binomtest(lost, lost + gained, 0.5).pvalue) if lost + gained else 1.0,
                                           "median_score_change": float(np.median(w["score"].values[sp] - o["score"].values[sp]))})
res["g2_flips"] = flips

# 5. conformal set composition of the proposed model, A9 nonconformity
comp = {}
for sd, df in runs[PROPOSED].items():
    val, cal = df[df["set"] == "val"], df[df["set"] == "cal"]
    T = fit_temperature(val["label"].values, val["score"].values)
    for al in (0.01, 0.05):
        zc = cal["score"].values / T; yc = cal["label"].values
        q = float(quantile(-(2 * yc - 1) * zc, al))
        for s in SETS:
            d = per_set(df, s); z = d["score"].values / T; y = d["label"].values
            in1, in0 = -z <= q, z <= q
            c = {"q": q, "both": float((in1 & in0).mean()), "empty": float((~in1 & ~in0).mean()),
                 "correct_single": float(((in1 ^ in0) & np.where(y == 1, in1, in0)).mean()),
                 "wrong_single": float(((in1 ^ in0) & ~np.where(y == 1, in1, in0)).mean())}
            comp.setdefault(s, {}).setdefault(str(al), []).append(c)
res["conformal_composition"] = comp
(ROOT / "results/extra_analysis.json").write_text(json.dumps(res, indent=1, default=float))
print("written results/extra_analysis.json")
for s in SETS:
    print(s, {n: (round(100 * v["mean"], 2), round(100 * v["lo"], 2), round(100 * v["hi"], 2)) for n, v in cis[s].items()})
    for r in tests[s]:
        print("   ", r["baseline"], r["seeds"], round(100 * r["diff"], 2), [round(100 * r["lo"], 2), round(100 * r["hi"], 2)],
              "p", round(r["p_boot_holm"], 4), "McN", r["mcnemar_b"], r["mcnemar_c"], round(r["p_mcnemar_holm"], 4))
