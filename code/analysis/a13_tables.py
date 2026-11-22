"""Tables of the target-ham adaptation (A13) from results_a13.json and the saved MuST-Guard fusion gates."""
import json, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parents[2]
STUDY = HERE
res = json.load(open(HERE / "results/results_a13.json"))
P, SD, CI = res["point"], res["seed_sd"], res["ci"]
NAME = {"MultinomialNB": "Multinomial naive Bayes", "LR": "Logistic regression", "LinearSVM": "Linear SVM",
        "BERT-base": "BERT-base", "DistilBERT": "DistilBERT", "MuST-Guard": "MuST-Guard (ours)"}
f = lambda x: f"{100 * x:.1f}"
rows = []
for m, n in NAME.items():
    b, a = P[f"{m}|base"], P[f"{m}|A13"]
    rows.append(f"{n} & {f(b['TPR_all'])} & {f(a['TPR_all'])} $\\pm$ {f(SD[m + '|A13'])} & {f(a['TPR_enron_spam'])} & "
                f"{f(a['TPR_spamassassin'])} & {100 * a['FPR_all']:.2f} & {f(a['TPR_S1'])} & {f(a['TPR_S3'])} \\\\")
(HERE / "results/tables/tab_a13.tex").write_text("\n".join(rows) + "\n")
# fusion gates of MuST-Guard, mean over seeds, full model and A13 variant
R = STUDY / "results"
src = {"full": "mustguard/confirm/s{s}/out/scores_mg-deberta-v3-small_s{s}_best.parquet",
       "a13": "a13/a13-mg/out/scores_a13-deberta-v3-small_s{s}_best.parquet"}
g = {}
for k, p in src.items():
    d = pd.concat([pd.read_parquet(R / p.format(s=s)) for s in range(5)])
    d["corpus"] = d.msg_id.str.split("/").str[0]
    d = d[d.set.isin(["s1_test", "s2_test"])]
    d["grp"] = np.where(d.set == "s1_test", "S1", np.where(d.corpus == "enron_spam", "S2 Enron-Spam", "S2 SpamAssassin"))
    g[k] = d.groupby(["grp", "label"])[["gate_body", "gate_header", "gate_url", "gate_struct"]].mean()
rows = []
for grp in ["S1", "S2 Enron-Spam", "S2 SpamAssassin"]:
    for lab, ln in ((0, "ham"), (1, "spam")):
        x, y = g["full"].loc[(grp, lab)], g["a13"].loc[(grp, lab)]
        rows.append(f"{grp} {ln} & " + " & ".join(f"{v:.2f}" for v in x.values) + " & " + f"{y.gate_body:.2f} & {y.gate_struct:.2f} \\\\")
(HERE / "results/tables/tab_gate.tex").write_text("\n".join(rows) + "\n")
print(open(HERE / "results/tables/tab_a13.tex").read()); print(open(HERE / "results/tables/tab_gate.tex").read())
print({k: [round(100 * v, 1) for v in CI[k]] for k in CI if k.startswith(("MuST", "Distil"))})
