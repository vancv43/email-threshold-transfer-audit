"""G2 stress test of PROTOCOL.md A5 for the classical baselines.

Per seed, each model is refitted with its configuration selected in classical_seed<seed>.json, then scored on
S3 ham + kept G2 rewrites and on S3 ham + the original versions of the same spam.
Usage: python classical_g2.py <seed>   Output: results/baselines/classical_g2_seed<seed>.json
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, load_messages, split_frame, threshold_at_fpr, tpr_at_fpr

SEED = int(sys.argv[1])
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
S = split_frame(df, SEED)
s3 = S["s3_test"]
g2 = pd.read_parquet(ROOT / "data/rewrites/g2/rewrite_g2_kept.parquet")
orig = s3.set_index("msg_id").loc[g2["msg_id"]]
ham = s3[s3["label"] == 0]
rewritten_text = orig["subject"].values + "\n" + g2["rewritten"].str[:5000].values
sets = {"g2_rewrite": pd.concat([ham["text"], pd.Series(rewritten_text)], ignore_index=True),
        "g2_original": pd.concat([ham["text"], orig["text"].reset_index(drop=True)], ignore_index=True)}
y = np.r_[np.zeros(len(ham), int), np.ones(len(g2), int)]

prev = json.loads((ROOT / f"results/baselines/classical_seed{SEED}.json").read_text())["models"]
res = {"seed": SEED, "n_ham": int(len(ham)), "n_spam": int(len(g2)), "models": {}}
vecs = {}
for mname, make in MODELS.items():
    sel = prev[mname]["selected"]
    v = sel["view"]
    if v not in vecs:
        vec = TfidfVectorizer(**VIEWS[v]).fit(S["train"]["text"])
        vecs[v] = (vec, vec.transform(S["train"]["text"]))
    vec, Xtr = vecs[v]
    m = make(sel["param"]).fit(Xtr, S["train"]["label"].values)
    s_cal = score(m, vec.transform(S["cal"]["text"]))
    thr = threshold_at_fpr(S["cal"]["label"].values, s_cal, 0.001)
    s3_check = tpr_at_fpr(s3["label"].values, score(m, vec.transform(s3["text"])), 0.001)
    out = {"s3_tpr_fpr_0p1_refit": s3_check, "s3_tpr_fpr_0p1_stored": prev[mname]["test"]["s3"]["tpr_fpr_0p1"]}
    for name, txt in sets.items():
        s = score(m, vec.transform(txt))
        out[name] = {"tpr_fpr_0p1": tpr_at_fpr(y, s, 0.001), "tpr_fpr_1": tpr_at_fpr(y, s, 0.01),
                     "cal_thr_tpr": float((s[y == 1] > thr).mean())}
    out["drop_tpr_fpr_0p1"] = out["g2_original"]["tpr_fpr_0p1"] - out["g2_rewrite"]["tpr_fpr_0p1"]
    res["models"][mname] = out
    print(mname, json.dumps(out), flush=True)
(ROOT / f"results/baselines/classical_g2_seed{SEED}.json").write_text(json.dumps(res, indent=1))
