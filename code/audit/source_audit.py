"""Source-shortcut audit of PROTOCOL.md A1: can one view alone tell which corpus a message came from?

For each view, a logistic regression predicts the corpus within one class (ham sources, spam sources), with
5-fold cross-validation grouped by near-duplicate cluster. A macro one-vs-rest AUC above 0.9 flags the view.
A second check predicts the label from each view on the training pool, which shows how much a view alone carries.
Output: results/audit/source_audit.json
"""
import json, sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, load_messages

PER_SOURCE = 3000
df, hdr = load_messages()
df["header_text"] = hdr.map(lambda h: " ".join(f"{k}={v}" for k, v in h.items() if k != "subject")
                            + " " + h.get("subject", ""))
df["url_text"] = df["urls"].map(lambda u: " ".join(json.loads(u)))
struct = pd.DataFrame(df["struct"].map(json.loads).tolist()).fillna(0)
df["n_urls"] = df["urls"].map(lambda u: len(json.loads(u)))


def view_matrix(name, d):
    if name == "struct":
        X = np.log1p(struct.loc[d.index].assign(n_urls=d["n_urls"]).values.astype(float))
        return sparse.csr_matrix(X)
    col = {"body": "text", "header": "header_text", "url": "url_text"}[name]
    pat = r"(?u)\b\w+\b" if name != "url" else r"[A-Za-z0-9]+"
    vec = TfidfVectorizer(token_pattern=pat, min_df=2, max_features=100000, sublinear_tf=True)
    try:
        return vec.fit_transform(d[col])
    except ValueError:  # empty vocabulary
        return sparse.csr_matrix((len(d), 1))


def cv_auc(X, y, groups):
    classes = np.unique(y)
    prob = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te]) if len(m.classes_) == len(classes) else 0.5
    if len(classes) == 2:
        return float(roc_auc_score(y, prob[:, 1]))
    return float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))


rng = np.random.RandomState(0)
out = {"per_source_cap": PER_SOURCE, "flag_threshold": 0.9, "source_within_class": {}, "label_from_view_pool": {}}
for label, name in [(0, "ham"), (1, "spam")]:
    d = df[df["label"] == label]
    d = d.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), PER_SOURCE), random_state=rng))
    if d["source"].nunique() < 2:
        continue
    y = d["source"].astype("category").cat.codes.values
    res = {"sources": d["source"].value_counts().to_dict()}
    for v in ["body", "header", "url", "struct"]:
        res[v] = cv_auc(view_matrix(v, d), y, d["dup_cluster"].values)
    out["source_within_class"][name] = res
    print(name, res, flush=True)

pool = df[df["source"].isin(["untroubled", "nazario", "apache", "gnu", "python", "fedora"]) & df["year"].between(2015, 2020)]
pool = pool.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 2 * PER_SOURCE), random_state=rng))
for v in ["body", "header", "url", "struct"]:
    out["label_from_view_pool"][v] = cv_auc(view_matrix(v, pool), pool["label"].values, pool["dup_cluster"].values)
print("label_from_view_pool", out["label_from_view_pool"], flush=True)
(ROOT / "results/audit").mkdir(parents=True, exist_ok=True)
(ROOT / "results/audit/source_audit.json").write_text(json.dumps(out, indent=1))
