"""Analysis of PROTOCOL.md A11 from saved per-message scores only (no training).

Proposed = MuST-Guard v2 (results/a11/jnsm-a11-mg2-s*). Compared with MuST-Guard v1 (results/mustguard/confirm/), the
transformer baselines (results/gpu7/), DistilBERT under the T1 rule (results/a11/jnsm-a11-distil-s*), and the classical
baselines under the v1 rule (results/a10/) and the T1 rule (results/a11/jnsm-a11-classical/).
Per run and test set: TPR at FPR 0.1% and the FPR realized at the calibration threshold.
Paired test as in extra_analysis.py: stratified per-message bootstrap of the seed-mean TPR at FPR 0.1%, 10,000 resamples,
one resample shared by all seeds and models on S2, S3 and G2, per seed on S1; Holm across comparators per test set.
Output: results/a11/summary_a11.json
"""
import glob, json, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, threshold_at_fpr, tpr_at_fpr

B, RNG = 10000, np.random.default_rng(20261115)
SETS = ["s1_test", "s3_test", "g2_rewrite", "s2_test"]
PROPOSED = "MuST-Guard v2"
GPU = {"bert-base-cased": "BERT-base, earlier pipeline", "distilbert-base-uncased": "DistilBERT",
       "roberta-base": "RoBERTa-base", "deberta-v3-base": "DeBERTa-v3-base"}
CLASSICAL = {"MultinomialNB": "Multinomial naive Bayes", "LR": "Logistic regression", "LinearSVM": "Linear SVM"}
seed_of = lambda f: int(re.search(r"_s(\d)_best", f).group(1))


def load():
    runs = {}
    for key, name in GPU.items():
        for f in glob.glob(str(ROOT / f"results/gpu7/jnsm-tb7-{key}-s*/out/scores_*_best.parquet")):
            runs.setdefault(name, {})[seed_of(f)] = pd.read_parquet(f)
    for f in glob.glob(str(ROOT / "results/mustguard/confirm/s*/out/scores_*_best.parquet")):
        runs.setdefault("MuST-Guard v1", {})[seed_of(f)] = pd.read_parquet(f)
    for f in glob.glob(str(ROOT / "results/a11/jnsm-a11-distil-s*/out/scores_*_best.parquet")):
        runs.setdefault("DistilBERT, T1 rule", {})[seed_of(f)] = pd.read_parquet(f)
    for f in glob.glob(str(ROOT / "results/a11/jnsm-a11-mg2-s*/out/scores_*_best.parquet")):
        runs.setdefault(PROPOSED, {})[seed_of(f)] = pd.read_parquet(f)
    for pat, tag in (("results/a10/classical_scores_seed*.parquet", ""),
                     ("results/a11/jnsm-a11-classical/out/classical/classical_t1_scores_seed*.parquet", ", T1 rule")):
        for f in glob.glob(str(ROOT / pat)):
            d = pd.read_parquet(f)
            for key, name in CLASSICAL.items():
                runs.setdefault(name + tag, {})[int(d["seed"].iloc[0])] = d[d["model"] == key].drop(columns=["model", "seed"])
    return runs


def tpr_fast(ham, spam, fpr=0.001):
    k = int(np.floor(fpr * len(ham)))
    t = np.partition(ham, len(ham) - 1 - k)[len(ham) - 1 - k]
    return float((spam > t).mean())


def per_set(df, s):
    return df[df["set"] == s].set_index("msg_id")


def split(d):
    y, sc = d["label"].values, d["score"].values
    return sc[y == 0], sc[y == 1]


runs = load()
assert sorted(runs[PROPOSED]) == list(range(5)), sorted(runs[PROPOSED])
res = {"runs": {n: sorted(r) for n, r in runs.items()}, "per_seed": {}, "selected": {}}

# per-run point metrics, with the same test messages as the proposed run of that seed
for name, rs in runs.items():
    for sd, df in rs.items():
        cal = df[df["set"] == "cal"]; thr = threshold_at_fpr(cal["label"].values, cal["score"].values, 0.001)
        for s in SETS:
            d = per_set(df, s)
            ref = per_set(runs[PROPOSED][sd], s)
            assert set(d.index) == set(ref.index), (name, sd, s)
            h, p = split(d)
            assert abs(tpr_fast(h, p) - tpr_at_fpr(d["label"].values, d["score"].values, 0.001)) < 1e-9, (name, sd, s)
            res["per_seed"].setdefault(name, {}).setdefault(s, {})[sd] = {"tpr_fpr_0p1": tpr_fast(h, p),
                                                                        "cal_thr_fpr": float((h > thr).mean())}
for f in sorted(glob.glob(str(ROOT / "results/a11/jnsm-a11-*-s*/out/selected_*.json"))):
    j = json.load(open(f)); res["selected"][f.split("/")[-1][9:-5]] ={k: j.get(k) for k in ("index", "config", "sel_key")}

# paired bootstrap, proposed minus each comparator on common seeds
tests = {}
for s in SETS:
    ref = {sd: per_set(runs[PROPOSED][sd], s) for sd in runs[PROPOSED]}
    arr = {(n, sd): split(per_set(df, s).loc[ref[sd].index]) for n, rs in runs.items() for sd, df in rs.items()}
    draws = {k: np.empty(B) for k in arr}
    for b0 in range(0, B, 500):
        nb = min(500, B - b0); idx = {}
        for sd in sorted(ref):
            y = ref[sd]["label"].values; nh, ns = int((y == 0).sum()), int((y == 1).sum())
            idx[sd] = next(iter(idx.values())) if (s != "s1_test" and idx) else \
                (RNG.integers(0, nh, (nb, nh)), RNG.integers(0, ns, (nb, ns)))
        for (n, sd), (h, p) in arr.items():
            ih, ip = idx[sd]
            for j in range(nb):
                draws[(n, sd)][b0 + j] = tpr_fast(h[ih[j]], p[ip[j]])
    rows = []
    for n in runs:
        if n == PROPOSED:
            continue
        common = sorted(set(runs[n]) & set(runs[PROPOSED]))
        diff = np.mean([draws[(PROPOSED, sd)] - draws[(n, sd)] for sd in common], axis=0)
        pt = np.mean([tpr_fast(*arr[(PROPOSED, sd)]) - tpr_fast(*arr[(n, sd)]) for sd in common])
        p = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / B))
        rows.append({"comparator": n, "seeds": common, "diff": float(pt), "lo": float(np.percentile(diff, 2.5)),
                     "hi": float(np.percentile(diff, 97.5)), "p_boot": p})
    o = np.argsort([r["p_boot"] for r in rows]); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (len(rows) - rank) * rows[i]["p_boot"])); rows[i]["p_boot_holm"] = run
    tests[s] = rows
    print("bootstrap done", s, flush=True)
res["paired_tests"] = tests
(ROOT / "results/a11/summary_a11.json").write_text(json.dumps(res, indent=1))

for s in SETS:
    print(f"\n== {s}")
    for n in runs:
        v = [res["per_seed"][n][s][sd]["tpr_fpr_0p1"] for sd in sorted(runs[n])]
        r = res["per_seed"][n][s]; fp = [r[sd]["cal_thr_fpr"] for sd in sorted(runs[n])]
        t = next((x for x in tests[s] if x["comparator"] == n), None)
        print(f"{n:34s} {np.mean(v):.3f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f} FPR {100 * np.mean(fp):5.1f}% "
              f"seeds {[round(x, 3) for x in v]}" + (f"  diff {t['diff']:+.3f} [{t['lo']:+.3f},{t['hi']:+.3f}] pHolm {t['p_boot_holm']:.4f}" if t else ""))
