"""Leave-one-source-out check on ham of PROTOCOL.md A7 for the classical baselines.

Per seed and held-out ham source h: ham of h is removed from train, val and cal; each model is tuned again on the
same 20-configuration grid by validation PR-AUC; the threshold is set on cal at FPR 0.1%; the test set is S3 with
h as the only ham source plus all S3 spam.
Usage: python classical_loso.py <seed>   Output: results/baselines/classical_loso_seed<seed>.json
"""
import json, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, load_messages, metrics, split_frame, threshold_at_fpr

SEED = int(sys.argv[1])
HAM_SOURCES = ["apache", "gnu", "python", "fedora"]
GRID = np.logspace(-3, 2, 10)
# same definitions as classical.py
VIEWS = {"word12": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True),
         "char35": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True)}
MODELS = {"MultinomialNB": lambda a: MultinomialNB(alpha=a),
          "LR": lambda c: LogisticRegression(C=c, max_iter=3000, solver="liblinear", random_state=SEED),
          "LinearSVM": lambda c: LinearSVC(C=c, max_iter=5000, random_state=SEED)}


def score(m, X):
    return m.predict_log_proba(X)[:, 1] - m.predict_log_proba(X)[:, 0] if hasattr(m, "predict_log_proba") \
        else m.decision_function(X)


df, _ = load_messages()
S0 = split_frame(df, SEED)
res = {"seed": SEED, "held_out": {}}
for h in HAM_SOURCES:
    drop = lambda d: d[~((d["label"] == 0) & (d["source"] == h))].reset_index(drop=True)
    S = {k: drop(S0[k]) for k in ["train", "val", "cal"]}
    t = S0["s3_test"]
    test = t[(t["label"] == 1) | (t["source"] == h)].reset_index(drop=True)
    X = {}
    for vname, kw in VIEWS.items():
        vec = TfidfVectorizer(**kw).fit(S["train"]["text"])
        X[vname] = {k: vec.transform(S[k]["text"]) for k in S} | {"test": vec.transform(test["text"])}
    ytr, yva, yca = (S[k]["label"].values for k in ["train", "val", "cal"])
    out = {"n_test_ham": int((test["label"] == 0).sum()), "n_test_spam": int(test["label"].sum()), "models": {}}
    for mname, make in MODELS.items():
        trials = []
        for vname in VIEWS:
            for g in GRID:
                m = make(g).fit(X[vname]["train"], ytr)
                trials.append((float(average_precision_score(yva, score(m, X[vname]["val"]))), vname, float(g), m))
        v_pr, v, g, m = max(trials, key=lambda r: r[0])
        s_cal = score(m, X[v]["cal"])
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(s_cal.reshape(-1, 1), yca)
        s = score(m, X[v]["test"])
        out["models"][mname] = {"selected": {"view": v, "param": g, "val_pr_auc": v_pr},
                                "test": metrics(test["label"].values, s, platt.predict_proba(s.reshape(-1, 1))[:, 1],
                                                threshold_at_fpr(yca, s_cal, 0.001))}
        print(SEED, h, mname, round(out["models"][mname]["test"]["tpr_fpr_0p1"], 4), flush=True)
    res["held_out"][h] = out
(ROOT / "results/baselines").mkdir(parents=True, exist_ok=True)
(ROOT / f"results/baselines/classical_loso_seed{SEED}.json").write_text(json.dumps(res, indent=1))
