"""Classical text baselines of PROTOCOL.md: MultinomialNB, LR and LinearSVM on TF-IDF.

Per seed: 20 configurations per model (2 TF-IDF views x 10 alpha or C values), selected by validation PR-AUC,
Platt scaling on the calibration split, then one evaluation on S1, S2 and S3 (static, also per test year).
Usage: python classical.py <seed>   Output: results/baselines/classical_seed<seed>.json
"""
import json, sys, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, load_messages, metrics, split_frame, threshold_at_fpr

SEED = int(sys.argv[1])
GRID = np.logspace(-3, 2, 10)
VIEWS = {"word12": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True),
         "char35": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True)}
MODELS = {"MultinomialNB": lambda a: MultinomialNB(alpha=a),
          "LR": lambda c: LogisticRegression(C=c, max_iter=3000, solver="liblinear", random_state=SEED),
          "LinearSVM": lambda c: LinearSVC(C=c, max_iter=5000, random_state=SEED)}


def score(m, X):
    return m.predict_log_proba(X)[:, 1] - m.predict_log_proba(X)[:, 0] if hasattr(m, "predict_log_proba") \
        else m.decision_function(X)


df, _ = load_messages()
S = split_frame(df, SEED)
tests = {"s1": S["s1_test"], "s2": S["s2_test"], "s3": S["s3_test"]}
res = {"seed": SEED, "sizes": {k: int(len(v)) for k, v in S.items()}, "models": {}}
X = {}
for vname, kw in VIEWS.items():
    t0 = time.time()
    vec = TfidfVectorizer(**kw).fit(S["train"]["text"])
    X[vname] = {k: vec.transform(S[k]["text"]) for k in ["train", "val", "cal"]}
    X[vname].update({k: vec.transform(v["text"]) for k, v in tests.items()})
    print(f"seed {SEED} view {vname} {X[vname]['train'].shape} {time.time() - t0:.0f}s", flush=True)

ytr, yva, yca = S["train"]["label"].values, S["val"]["label"].values, S["cal"]["label"].values
for mname, make in MODELS.items():
    trials = []
    for vname in VIEWS:
        for g in GRID:
            m = make(g).fit(X[vname]["train"], ytr)
            trials.append({"view": vname, "param": float(g),
                           "val_pr_auc": float(average_precision_score(yva, score(m, X[vname]["val"]))), "model": m})
    best = max(trials, key=lambda t: t["val_pr_auc"])
    m, v = best["model"], best["view"]
    s_cal = score(m, X[v]["cal"])
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(s_cal.reshape(-1, 1), yca)
    thr = threshold_at_fpr(yca, s_cal, 0.001)
    out = {"selected": {"view": v, "param": best["param"], "val_pr_auc": best["val_pr_auc"]},
           "trials": [{k: t[k] for k in ("view", "param", "val_pr_auc")} for t in trials], "test": {}}
    for tname, d in tests.items():
        s = score(m, X[v][tname])
        p = platt.predict_proba(s.reshape(-1, 1))[:, 1]
        out["test"][tname] = metrics(d["label"].values, s, p, thr)
        if tname == "s2":
            for src in sorted(d["source"].unique()):
                k = (d["source"] == src).values
                out["test"][f"s2_{src}"] = metrics(d["label"].values[k], s[k], p[k], thr)
        if tname == "s3":
            for y in sorted(d["year"].unique()):
                k = (d["year"] == y).values
                out["test"][f"s3_{y}"] = metrics(d["label"].values[k], s[k], p[k], thr)
    res["models"][mname] = out
    print(mname, out["selected"], {t: round(out["test"][t].get("tpr_fpr_0p1", -1), 4) for t in tests}, flush=True)

(ROOT / "results/baselines").mkdir(parents=True, exist_ok=True)
(ROOT / f"results/baselines/classical_seed{SEED}.json").write_text(json.dumps(res, indent=1))
