"""Tables of A14 from results_a14.json: ablation (A14a) and frozen-gate adaptation (A14b)."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parents[2]
r = json.load(open(HERE / "results/results_a14.json"))
m, sd = r["a14a"]["mean"], r["a14a"]["sd"]
f = lambda v, k: f"{100 * m[v][k]:.1f} $\\pm$ {100 * sd[v][k]:.1f}"
NAMES = [("bodynoaug", "Body view, no augmentation"), ("body", "Body view"), ("noaug", "Four views, no augmentation"),
         ("nostruct", "Without the structure view"), ("full", "MuST-Guard (ours)")]
rows = [f"{n} & {f(v, 's1_test')} & {f(v, 's3_test')} & {f(v, 'g2_rewrite')} & {100 * m[v]['g2_drop']:+.2f} & "
        f"{f(v, 's2_test')} & {f(v, 's2_fpr_tcal')} \\\\" for v, n in NAMES]
(HERE / "results/tables/tab_a14a.tex").write_text("\n".join(rows) + "\n")
b, bs = r["a14b"]["per_arm"], r["a14b"]["sd"]
ARMS = [("stage1", "Recalibrated only"), ("trainable", "Adapted, gate trainable"), ("frozen", "Adapted, gate frozen")]
rows = [f"{n} & {100 * b[a]['TPR']:.1f} $\\pm$ {100 * bs[a]:.1f} & {100 * b[a]['enron_spam']:.1f} & {100 * b[a]['spamassassin']:.1f} & "
        f"{100 * b[a]['FPR']:.2f} & {100 * b[a]['s1_test']:.1f} & {100 * b[a]['s3_test']:.1f} & {b[a]['gbody_enron_spam']:.2f} & "
        f"{b[a]['gstruct_enron_spam']:.2f} \\\\" for a, n in ARMS]
(HERE / "results/tables/tab_a14b.tex").write_text("\n".join(rows) + "\n")
print(open(HERE / "results/tables/tab_a14a.tex").read()); print(open(HERE / "results/tables/tab_a14b.tex").read())
print({k: (round(v["diff"] * 100, 1), round(v["p_holm"], 3)) for k, v in r["a14a"]["tests"]["s2_test"].items()})
print({k: (round(v["diff"] * 100, 2), round(v["p_holm"], 3)) for k, v in r["a14b"]["tests"].items()})
