"""A13 analysis: target-ham adaptation on all S2 evaluation messages, MuST-Guard after adaptation compared with the
full MuST-Guard and with every baseline before and after adaptation. Two-stage bootstrap over seeds and messages,
threshold re-estimated on the resampled target-calibration ham in every replicate, Holm across comparisons."""
import json, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parents[2]
STUDY = HERE
R = STUDY / "results"
GPU = {"MuST-Guard": ("mg/out/scores_a13-deberta-v3-small_s{s}_best.parquet", "mustguard/confirm/s{s}/out/scores_mg-deberta-v3-small_s{s}_best.parquet"),
       "BERT-base": ("bert/out/scores_a13-bert-base-cased_s{s}_best.parquet", "gpu7/tb-bert-base-cased-s{s}/out/scores_bert-base-cased_s{s}_best.parquet"),
       "DistilBERT": ("distil/out/scores_a13-distilbert-base-uncased_s{s}_best.parquet", "gpu7/tb-distilbert-base-uncased-s{s}/out/scores_distilbert-base-uncased_s{s}_best.parquet")}
CLS = ["MultinomialNB", "LR", "LinearSVM"]
SEEDS, B = range(5), 1000


def load(seed):
    """Per seed: tcal ham ids, eval ids, and {(model, arm): Series score by msg_id}."""
    out = {}
    for m, (a, b) in GPU.items():
        A = pd.read_parquet(R / "a13" / f"a13-{a.format(s=seed)}"); Bb = pd.read_parquet(R / b.format(s=seed))
        out[(m, "A13")] = A; out[(m, "base")] = Bb
    A = pd.read_parquet(R / f"a13/a13-classical/out/classical/classical_a13_scores_seed{seed}.parquet")
    Bb = pd.read_parquet(R / f"a10/classical_scores_seed{seed}.parquet")
    for m in CLS:
        out[(m, "A13")] = A[A.model == m]; out[(m, "base")] = Bb[Bb.model == m]
    ref = out[("DistilBERT", "A13")]
    tc = ref[ref.set == "tcal"].msg_id.values
    ev = ref[ref.set == "s2_test"][["msg_id", "label"]].reset_index(drop=True)
    s1 = ref[ref.set == "s1_test"][["msg_id", "label"]].reset_index(drop=True)
    s3 = ref[ref.set == "s3_test"][["msg_id", "label"]].reset_index(drop=True)
    sc = {}
    for k, d in out.items():
        d = d.drop_duplicates(["set", "msg_id"])
        g = lambda st, ids: d[d.set.isin(st)].drop_duplicates("msg_id").set_index("msg_id").score.reindex(ids).values
        sc[k] = dict(tcal=g(["tcal", "s2_test", "cal", "val"], tc), ev=g(["s2_test"], ev.msg_id),
                     s1=g(["s1_test"], s1.msg_id), s3=g(["s3_test"], s3.msg_id))
        assert not np.isnan(sc[k]["tcal"]).any() and not np.isnan(sc[k]["ev"]).any(), k
    return ev, s1, s3, sc


def thr(h):  # flag score > t keeps FPR <= 0.1% on h
    k = int(np.floor(0.001 * len(h)))
    return -np.partition(-h, k)[k]


def tpr_at(h, pos):
    return (pos > thr(h)).mean()


data = {s: load(s) for s in SEEDS}
SEL = set(); KEYS = list(data[0][3]); best = "MuST-Guard"
rows = []
for s in SEEDS:
    ev, s1, s3, sc = data[s]
    for part in ("report",):
        m_ = np.ones(len(ev), bool)
        for (m, arm) in KEYS:
            x = sc[(m, arm)]; t = thr(x["tcal"]); r = dict(model=m, arm=arm, seed=s, part=part)
            for c in ("all", "enron_spam", "spamassassin"):
                cm = m_ & ((ev.msg_id.str.split("/").str[0] == c).values if c != "all" else True)
                r[f"TPR_{c}"] = (x["ev"][cm & (ev.label == 1).values] > t).mean()
                r[f"FPR_{c}"] = (x["ev"][cm & (ev.label == 0).values] > t).mean()
            if part == "report":
                for n, d in (("S1", s1), ("S3", s3)):
                    y = d.label.values; r[f"TPR_{n}"] = tpr_at(x[n.lower()][y == 0], x[n.lower()][y == 1])
            rows.append(r)
df = pd.DataFrame(rows)
# two-stage bootstrap: MuST-Guard after adaptation vs every other model and arm
bs = np.random.default_rng(7)
diffs = {k: [] for k in KEYS if k != (best, "A13")}
boot_tpr = {k: [] for k in KEYS}
for b in range(B):
    seeds = bs.choice(list(SEEDS), 5)
    acc = {k: 0.0 for k in KEYS}
    for s in seeds:
        ev, _, _, sc = data[s]
        rep = ~ev.msg_id.isin(SEL).values
        sp = np.flatnonzero(rep & (ev.label == 1).values)
        it = bs.integers(0, len(sc[KEYS[0]]["tcal"]), len(sc[KEYS[0]]["tcal"])); ip = bs.choice(sp, len(sp))
        for k in KEYS:
            acc[k] += (sc[k]["ev"][ip] > thr(sc[k]["tcal"][it])).mean() / 5
    for k in KEYS:
        boot_tpr[k].append(acc[k])
    for k in diffs:
        diffs[k].append(acc[(best, "A13")] - acc[k])
rep = df[df.part == "report"]
point = rep.groupby(["model", "arm"]).mean(numeric_only=True)
res = {"reference": best, "point": {f"{m}|{a}": point.loc[(m, a)].drop("seed").to_dict() for m, a in point.index},
       "seed_sd": {"|".join(k): v for k, v in rep.groupby(["model", "arm"]).TPR_all.std().items()},
       "ci": {"|".join(k): np.percentile(v, [2.5, 97.5]).tolist() for k, v in boot_tpr.items()}, "tests": {}}
ps = {}
for k, v in diffs.items():
    v = np.array(v); ps[k] = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / B)
order = sorted(ps, key=ps.get); m = len(order); run = 0
for i, k in enumerate(order):
    run = max(run, min(1.0, (m - i) * ps[k]))
    v = np.array(diffs[k]); res["tests"]["|".join(k)] = dict(diff=float(v.mean()), lo=float(np.percentile(v, 2.5)),
                                                           hi=float(np.percentile(v, 97.5)), p=ps[k], p_holm=run)
json.dump(res, open(HERE / "results/results_a13.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print((point[["TPR_all", "FPR_all", "TPR_enron_spam", "TPR_spamassassin", "TPR_S1", "TPR_S3"]] * 100).round(1).to_string())
for k, t in res["tests"].items():
    print(k, {a: round(b * 100, 1) if a in ("diff", "lo", "hi") else round(b, 4) for a, b in t.items()})
