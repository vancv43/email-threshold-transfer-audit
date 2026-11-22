"""A11: the three classical baselines retuned with the T1 selection rule (same grid and views as classical.py).

Per seed: fit TF-IDF on train (held-out ham source removed), train the 20 configurations per model, choose by
TPR at FPR 0.1% on val spam plus the held-out ham (split ood_ham), ties by val PR-AUC, then score cal, ood_ham,
S1, S2, S3 and the two G2 sets once. Requires data/processed/splits.parquet already passed through t1_split.py.
Usage: python classical_t1.py <out_dir> [--seeds 0,1,2,3,4] [--sub N]   (--sub: smoke test on N rows per split)
"""
import argparse, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import ROOT, load_messages, split_frame, threshold_at_fpr, tpr_at_fpr  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--sub", type=int, default=0)
A = ap.parse_args()
OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
GRID = np.logspace(-3, 2, 10)
VIEWS = {"word12": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True),
         "char35": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000, sublinear_tf=True)}


def models(seed):
    return {"MultinomialNB": lambda a: MultinomialNB(alpha=a),
            "LR": lambda c: LogisticRegression(C=c, max_iter=3000, solver="liblinear", random_state=seed),
            "LinearSVM": lambda c: LinearSVC(C=c, max_iter=5000, random_state=seed)}


def score(m, X):
    return m.predict_log_proba(X)[:, 1] - m.predict_log_proba(X)[:, 0] if hasattr(m, "predict_log_proba") \
        else m.decision_function(X)


def seed_job(seed):
    t0 = time.time()
    df, _ = load_messages()
    S = split_frame(df, seed)
    s3 = S["s3_test"]
    g2 = pd.read_parquet(ROOT / "data/rewrites/g2/rewrite_g2_kept.parquet")
    orig = s3.set_index("msg_id").loc[g2["msg_id"]].reset_index()
    rew = orig.copy(); rew["text"] = orig["subject"].values + "\n" + g2["rewritten"].str[:5000].values
    ham = s3[s3["label"] == 0]
    S["g2_rewrite"] = pd.concat([ham, rew], ignore_index=True); S["g2_original"] = pd.concat([ham, orig], ignore_index=True)
    if A.sub:
        S = {k: d.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), A.sub // 2), random_state=0))
             .reset_index(drop=True) for k, d in S.items()}
    held = S["ood_ham"]["source"].unique()
    assert len(held) == 1 and not ((S["train"]["source"] == held[0]) & (S["train"]["label"] == 0)).any()
    sel_set = pd.concat([S["val"][S["val"]["label"] == 1], S["ood_ham"]], ignore_index=True)
    X = {}
    for v, kw in VIEWS.items():
        vec = TfidfVectorizer(**kw).fit(S["train"]["text"])
        X[v] = {k: vec.transform(d["text"]) for k, d in {**S, "sel": sel_set}.items()}
    res, rows = {"seed": seed, "holdout": held[0], "trials": {}, "selected": {}, "test": {}}, []
    yc = S["cal"]["label"].values
    for mname, make in models(seed).items():
        trials = []
        for v in VIEWS:
            for g in GRID:
                m = make(g).fit(X[v]["train"], S["train"]["label"].values)
                key = (tpr_at_fpr(sel_set["label"].values, score(m, X[v]["sel"]), 0.001),
                       float(average_precision_score(S["val"]["label"].values, score(m, X[v]["val"]))))
                trials.append({"view": v, "param": float(g), "key": key, "m": m})
        best = max(trials, key=lambda t: t["key"])  # ties: first in grid order
        res["trials"][mname] = [{k: t[k] for k in ("view", "param", "key")} for t in trials]
        res["selected"][mname] = {"view": best["view"], "param": best["param"], "sel_key": list(best["key"])}
        thr = threshold_at_fpr(yc, score(best["m"], X[best["view"]]["cal"]), 0.001)
        res["test"][mname] = {}
        for k in ["cal", "ood_ham", "s1_test", "s2_test", "s3_test", "g2_rewrite", "g2_original"]:
            s = score(best["m"], X[best["view"]][k]); y = S[k]["label"].values
            rows.append(pd.DataFrame({"model": mname, "seed": seed, "set": k, "msg_id": S[k]["msg_id"].values,
                                      "label": y, "score": s}))
            if k.startswith("s") or k.startswith("g2"):
                res["test"][mname][k] = {"tpr_fpr_0p1": tpr_at_fpr(y, s, 0.001),
                                         "cal_thr_fpr": float((s[y == 0] > thr).mean())}
    pd.concat(rows).to_parquet(OUT / f"classical_t1_scores_seed{seed}.parquet", index=False)
    (OUT / f"classical_t1_seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(f"seed {seed} done {time.time() - t0:.0f}s", json.dumps(res["selected"]), flush=True)
    return seed


if __name__ == "__main__":
    seeds = [int(s) for s in A.seeds.split(",")]
    with ProcessPoolExecutor(min(2, len(seeds))) as ex:
        done = list(ex.map(seed_job, seeds))
    if sorted(done) == seeds:
        open(OUT / "DONE", "w").write("ok")
