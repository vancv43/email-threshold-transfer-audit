"""A14 analysis (PROTOCOL.md A14). A14a: ablation variants against the full MuST-Guard of seeds 0-2, TPR at FPR 0.1%
read from each test curve, TPR and FPR at the calibration threshold on S2, and the G2 drop. A14b: stage 1, trainable
gate, and frozen gate, threshold at FPR 0.1% on the target-calibration ham, as in A13. Two-stage bootstrap over seeds
and messages with the threshold re-estimated in every replicate, Holm within each analysis."""
import json, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parents[1]
STUDY = HERE
R = STUDY / "results"
B = 1000


def thr(h):
    k = int(np.floor(0.001 * len(h)))
    return -np.partition(-h, k)[k]


def load(path):
    d = pd.read_parquet(path)
    d["corpus"] = d.msg_id.str.split("/").str[0]
    return d


def holm(ps):
    order = sorted(ps, key=ps.get); out, run = {}, 0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k])); out[k] = run
    return out


def boot(get, arms, ref, seeds, rng):
    """get(arm, seed, idx_h, idx_s) -> TPR; returns mean diffs ref-arm with CI and two-sided p."""
    diffs = {a: [] for a in arms if a != ref}
    for _ in range(B):
        ss = rng.choice(seeds, len(seeds)); acc = {a: 0.0 for a in arms}
        for s in ss:
            ih, isp = get(None, s, None, None)  # sizes
            jh, js = rng.integers(0, ih, ih), rng.integers(0, isp, isp)
            for a in arms:
                acc[a] += get(a, s, jh, js) / len(ss)
        for a in diffs:
            diffs[a].append(acc[ref] - acc[a])
    res = {}
    for a, v in diffs.items():
        v = np.array(v); p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / B)
        res[a] = dict(diff=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p=p)
    for a, p in holm({a: r["p"] for a, r in res.items()}).items():
        res[a]["p_holm"] = p
    return res


out = {}
# ---------- A14b ----------
ARMS = ["stage1", "trainable", "frozen"]
fz = {(a, s): load(R / f"a14/jnsm-a14-frozen/out/scores_a14b-{a}_s{s}_best.parquet") for a in ARMS for s in range(5)}
rows = []
for (a, s), d in fz.items():
    t = thr(d[d.set == "tcal"].score.values); e = d[d.set == "s2_test"]
    r = dict(arm=a, seed=s, TPR=(e[e.label == 1].score > t).mean(), FPR=(e[e.label == 0].score > t).mean())
    for c in ("enron_spam", "spamassassin"):
        r[c] = (e[(e.label == 1) & (e.corpus == c)].score > t).mean()
    for n in ("s1_test", "s3_test"):
        x = d[d.set == n]; r[n] = (x[x.label == 1].score > thr(x[x.label == 0].score.values)).mean()
    for c in ("enron_spam", "spamassassin"):
        g = e[(e.label == 1) & (e.corpus == c)]; r[f"gbody_{c}"] = g.gate_body.mean(); r[f"gstruct_{c}"] = g.gate_struct.mean()
    rows.append(r)
fb = pd.DataFrame(rows)
cache = {}
for (a, s), d in fz.items():
    cache[(a, s)] = (d[d.set == "tcal"].score.values, d[(d.set == "s2_test") & (d.label == 1)].score.values)
assert all((fz[("stage1", s)][fz[("stage1", s)].set == "tcal"].msg_id.values == fz[(a, s)][fz[(a, s)].set == "tcal"].msg_id.values).all() for a in ARMS for s in range(5))


def getb(a, s, jh, js):
    if a is None:
        return len(cache[("stage1", s)][0]), len(cache[("stage1", s)][1])
    h, sp = cache[(a, s)]
    return (sp[js] > thr(h[jh])).mean()


out["a14b"] = {"per_arm": {a: fb[fb.arm == a].drop(columns=["arm", "seed"]).mean().to_dict() for a in ARMS},
               "sd": {a: float(fb[fb.arm == a].TPR.std()) for a in ARMS},
               "per_seed": fb.to_dict("records"),
               "tests": boot(getb, ARMS, "frozen", list(range(5)), np.random.default_rng(14))}
# ---------- A14a ----------
VAR = ["full", "noaug", "nostruct", "body", "bodynoaug"]
def path(v, s):
    return (R / f"mustguard/confirm/s{s}/out/scores_mg-deberta-v3-small_s{s}_best.parquet" if v == "full"
            else R / f"a14/jnsm-a14-abl/out/scores_a14a-{v}_s{s}_best.parquet")
ab = {(v, s): load(path(v, s)) for v in VAR for s in range(3)}
rows, cache2 = [], {}
for (v, s), d in ab.items():
    r = dict(variant=v, seed=s)
    for n in ("s1_test", "s3_test", "s2_test", "g2_original", "g2_rewrite"):
        x = d[d.set == n]; r[n] = (x[x.label == 1].score > thr(x[x.label == 0].score.values)).mean()
    tc = thr(d[(d.set == "cal") & (d.label == 0)].score.values); e = d[d.set == "s2_test"]
    r["s2_tpr_tcal"] = (e[e.label == 1].score > tc).mean(); r["s2_fpr_tcal"] = (e[e.label == 0].score > tc).mean()
    r["g2_drop"] = r["g2_original"] - r["g2_rewrite"]
    rows.append(r)
    for n in ("s2_test", "g2_rewrite", "s3_test"):
        x = d[d.set == n]; cache2[(v, s, n)] = (x[x.label == 0].score.values, x[x.label == 1].score.values)
fa = pd.DataFrame(rows)
out["a14a"] = {"mean": fa.groupby("variant").mean(numeric_only=True).drop(columns="seed").to_dict("index"),
               "sd": fa.groupby("variant").std(numeric_only=True).drop(columns="seed").to_dict("index"),
               "per_seed": fa.to_dict("records"), "tests": {}}
for n in ("s2_test", "g2_rewrite", "s3_test"):
    def geta(v, s, jh, js, n=n):
        if v is None:
            return len(cache2[("full", s, n)][0]), len(cache2[("full", s, n)][1])
        h, sp = cache2[(v, s, n)]
        return (sp[js] > thr(h[jh])).mean()
    out["a14a"]["tests"][n] = boot(geta, VAR, "full", [0, 1, 2], np.random.default_rng(15))
json.dump(out, open(HERE / "results/results_a14.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print("A14b\n", (fb.groupby("arm")[["TPR", "FPR", "enron_spam", "spamassassin", "s1_test", "s3_test", "gbody_enron_spam", "gstruct_enron_spam"]].mean() * 100).round(2))
print({a: {k: round(v * 100 if k in ("diff", "lo", "hi") else v, 3) for k, v in t.items()} for a, t in out["a14b"]["tests"].items()})
print("A14a\n", (fa.groupby("variant")[["s1_test", "s3_test", "g2_original", "g2_rewrite", "g2_drop", "s2_test", "s2_tpr_tcal", "s2_fpr_tcal"]].mean() * 100).round(2))
for n, t in out["a14a"]["tests"].items():
    print(n, {a: {k: round(v * 100 if k in ("diff", "lo", "hi") else v, 3) for k, v in r.items()} for a, r in t.items()})
