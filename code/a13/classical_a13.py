"""A13: the three classical baselines retrained once at the configuration each seed selected, on the training split
plus the 5,000 adaptation ham; scores of tcal, S1, S2 (evaluation part) and S3. Needs splits passed through
a13_split.py and configs.json. Usage: python classical_a13.py <out_dir> <configs.json> [--seeds 0,1,2,3,4] [--sub N]
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_messages, split_frame  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("configs"); ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--sub", type=int, default=0)
A = ap.parse_args()
OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
CFG = json.load(open(A.configs))["classical"]
VIEWS = {"word12": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True),
         "char35": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True)}
MAKE = {"MultinomialNB": lambda a, s: MultinomialNB(alpha=a),
        "LR": lambda c, s: LogisticRegression(C=c, max_iter=3000, solver="liblinear", random_state=s),
        "LinearSVM": lambda c, s: LinearSVC(C=c, max_iter=5000, random_state=s)}


def score(m, X):
    return m.predict_log_proba(X)[:, 1] - m.predict_log_proba(X)[:, 0] if hasattr(m, "predict_log_proba") else m.decision_function(X)


df, _ = load_messages()
for seed in map(int, A.seeds.split(",")):
    S = split_frame(df, seed)
    if A.sub:
        S = {k: v.groupby("label", group_keys=False).apply(lambda g: g.head(A.sub // 2)).reset_index(drop=True) for k, v in S.items()}
    rows = []
    for name, c in CFG[str(seed)].items():
        vec = TfidfVectorizer(**VIEWS[c["view"]]).fit(S["train"]["text"])
        m = MAKE[name](c["param"], seed).fit(vec.transform(S["train"]["text"]), S["train"]["label"].values)
        for k in ["tcal", "s1_test", "s2_test", "s3_test"]:
            d = S[k]; rows.append(pd.DataFrame({"model": name, "seed": seed, "set": k, "msg_id": d["msg_id"].values,
                                                "label": d["label"].values, "score": score(m, vec.transform(d["text"]))}))
    pd.concat(rows).to_parquet(OUT / f"classical_a13_scores_seed{seed}.parquet", index=False)
    print("seed", seed, "done", flush=True)
open(OUT / "DONE", "w").write("ok")
