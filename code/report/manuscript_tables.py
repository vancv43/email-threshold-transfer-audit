"""Every result table of the manuscript, generated from the saved result files (PROTOCOL.md section 6).

Inputs: classical baselines (results/baselines/*.json), transformer baselines (scores in results/gpu7/), MuST-Guard
(scores in results/mustguard/confirm/ and results/mustguard/summary_confirm_a9.json), source audit, rewrite filter.
A model with fewer than five finished seeds is shown with its seed count; a model with none gets \\TBD.
Output: results/tables/*.tex (one table body each) and results/manuscript_numbers.json.
Usage: python manuscript_tables.py
"""
import glob, json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import ROOT, threshold_at_fpr, tpr_at_fpr

OUT = ROOT / "results/tables"; OUT.mkdir(parents=True, exist_ok=True)
SC = {"s1": "In-distribution", "s2": "Cross-corpus", "s3": "Temporal"}
CLASSICAL = [("MultinomialNB", "Multinomial naive Bayes"), ("LR", "Logistic regression"), ("LinearSVM", "Linear SVM")]
TRANSFORMERS = [("bert-base-cased", "BERT-base, earlier pipeline"), ("distilbert-base-uncased", "DistilBERT"),
                ("roberta-base", "RoBERTa-base"), ("deberta-v3-base", "DeBERTa-v3-base")]
PROPOSED = "Proposed"
num = {}


def ms(v, pct=True, d=1):
    """mean +- sd of a list, in percent by default; one value has no sd."""
    if not v:
        return "\\TBD"
    f = 100 if pct else 1
    m = f * float(np.mean(v))
    return f"{m:.{d}f}" if len(v) == 1 else f"{m:.{d}f} $\\pm$ {f * float(np.std(v, ddof=1)):.{d}f}"


def at_thr(y, s, thr):
    p = s > thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / max(1, int((y == 1).sum()))
    return {"accuracy": float((p == y).mean()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "fpr": fp / max(1, int((y == 0).sum()))}


def score_runs(files):
    """per-seed metrics of runs that saved scores (transformer baselines and the proposed model)."""
    runs = []
    for f in sorted(files):
        g = {k: v for k, v in pd.read_parquet(f).groupby("set")}
        thr = threshold_at_fpr(g["cal"]["label"].values, g["cal"]["score"].values, 0.001)
        r = {}
        for s in SC:
            y, sc = g[f"{s}_test"]["label"].values, g[f"{s}_test"]["score"].values
            r[s] = {"tpr": tpr_at_fpr(y, sc, 0.001), "auc": float(roc_auc_score(y, sc)), **at_thr(y, sc, thr)}
        for k in ("g2_original", "g2_rewrite"):
            y, sc = g[k]["label"].values, g[k]["score"].values
            r[k] = {"tpr": tpr_at_fpr(y, sc, 0.001), "thr_tpr": float((sc[y == 1] > thr).mean())}
        runs.append(r)
    return runs


models = {}  # display name -> list of per-seed dicts with the keys of score_runs
for key, name in CLASSICAL:
    rs = []
    for sd in range(5):
        t = json.load(open(ROOT / f"results/baselines/classical_seed{sd}.json"))["models"][key]["test"]
        g2 = json.load(open(ROOT / f"results/baselines/classical_g2_seed{sd}.json"))
        g2 = g2["models"][key]
        r = {}
        for s in SC:
            x = t[s]; n_ham = x["n"] - x["n_spam"]
            tp, fp = x["cal_thr_tpr"] * x["n_spam"], x["cal_thr_fpr"] * n_ham
            prec = tp / (tp + fp) if tp + fp else 0.0
            r[s] = {"tpr": x["tpr_fpr_0p1"], "auc": x["roc_auc"], "accuracy": (tp + n_ham - fp) / x["n"],
                    "precision": prec, "recall": x["cal_thr_tpr"], "fpr": x["cal_thr_fpr"],
                    "f1": 2 * prec * x["cal_thr_tpr"] / (prec + x["cal_thr_tpr"]) if prec + x["cal_thr_tpr"] else 0.0}
        r["g2_original"] = {"tpr": g2["g2_original"]["tpr_fpr_0p1"]}
        r["g2_rewrite"] = {"tpr": g2["g2_rewrite"]["tpr_fpr_0p1"]}
        rs.append(r)
    models[name] = rs
for key, name in TRANSFORMERS:
    models[name] = score_runs(glob.glob(str(ROOT / f"results/gpu7/tb-{key}-s*/out/scores_*_best.parquet")))
models[PROPOSED] = score_runs(glob.glob(str(ROOT / "results/mustguard/confirm/s*/out/scores_*_best.parquet")))
assert len(models[PROPOSED]) == 5


def col(name, *path):
    out = []
    for r in models[name]:
        v = r
        for p in path:
            v = v[p]
        out.append(v)
    return out


def nm(name):
    """display name; models with fewer than five finished seeds carry a dagger"""
    return name + ("$^\\dagger$" if 0 < len(models[name]) < 5 else "")


# Table: primary metric under the three scenarios
rows = []
for name in models:
    n = len(models[name])
    rows.append(f"{name} & {n if n else chr(92) + 'TBD'} & " + " & ".join(ms(col(name, s, "tpr")) if n else "\\TBD" for s in SC) + " \\\\")
    num[f"main/{name}"] = {s: col(name, s, "tpr") for s in SC}
(OUT / "tab_main.tex").write_text("\n".join(rows) + "\n")

# Table: operating point, threshold set on the calibration split at FPR 0.1%
rows = []
for s, sname in SC.items():
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{sname}}}}} \\\\")
    for name in models:
        if not models[name]:
            rows.append(f"\\quad {name} & " + " & ".join(["\\TBD"] * 6) + " \\\\"); continue
        rows.append(f"\\quad {nm(name)} & " + " & ".join(ms(col(name, s, k)) for k in ("accuracy", "precision", "recall", "f1"))
                    + f" & {ms(col(name, s, 'auc'), pct=False, d=3)} & {ms(col(name, s, 'fpr'))} \\\\")
(OUT / "tab_oppoint.tex").write_text("\n".join(rows) + "\n")

# Table: G2 stress set
rows = []
for name in models:
    if not models[name]:
        rows.append(f"{name} & \\TBD & \\TBD & \\TBD \\\\"); continue
    o, w = np.array(col(name, "g2_original", "tpr")), np.array(col(name, "g2_rewrite", "tpr"))
    rows.append(f"{nm(name)} & {ms(list(o))} & {ms(list(w))} & {ms(list(o - w), d=2)} \\\\")
    num[f"g2/{name}"] = {"original": list(o), "rewrite": list(w)}
(OUT / "tab_g2.tex").write_text("\n".join(rows) + "\n")

# Table: calibrated abstention of the proposed model, A9 nonconformity (A8 coverage kept for reference)
a9 = json.load(open(ROOT / "results/mustguard/summary_confirm_a9.json"))
rows = []
for k, kname in [("s1_test", "In-distribution"), ("s2_test", "Cross-corpus"), ("s3_test", "Temporal"),
                 ("g2_rewrite", "$G_2$ rewrites")]:
    for al in (0.01, 0.05):
        pr = [r[k] for r in a9["per_run"]]
        get = lambda c, m: [p[f"{c}_{al}"][m] for p in pr if p[f"{c}_{al}"][m] is not None]
        rows.append(f"{kname if al == 0.01 else ''} & {1 - al:.2f} & {ms(get('conformal_logit', 'coverage'))} & "
                    f"{ms(get('conformal_logit', 'abstain'))} & {ms(get('conformal_logit', 'accepted_tpr'))} & "
                    f"{ms(get('conformal_logit', 'accepted_fpr'), d=2)} & {ms(get('conformal', 'coverage'))} \\\\")
        num[f"conformal/{k}/{al}"] = {"a9_coverage": get("conformal_logit", "coverage"), "a8_coverage": get("conformal", "coverage")}
(OUT / "tab_conformal.tex").write_text("\n".join(rows) + "\n")

# Table: fusion gate weights of the proposed model, mean over seeds, spam and ham separately
rows = []
for k, kname in [("s1_test", "In-distribution"), ("s3_test", "Temporal"), ("s2_test", "Cross-corpus")]:
    for lab, lname in (("1", "spam"), ("0", "ham")):
        v = [[r[k]["gates"][f"gate_{b}"][lab] for r in a9["per_run"]] for b in ("body", "header", "url", "struct")]
        rows.append(f"{kname if lab == '1' else ''} & {lname} & " + " & ".join(ms(x, pct=False, d=2) for x in v) + " \\\\")
(OUT / "tab_gates.tex").write_text("\n".join(rows) + "\n")

# Table: source-shortcut audit; the A10 rerun on the six training-pool sources with the model's views, when present
au = json.load(open(ROOT / "results/audit/source_audit.json"))
A10 = ROOT / "results/a10"
au10 = json.load(open(A10 / "a10_audit.json")) if (A10 / "a10_audit.json").exists() else None
VN = {"body": "Body", "header": "Header", "url": "Links", "struct": "Structure"}
rows = ["\\multicolumn{4}{@{}l}{\\emph{All eight sources, audit views}} \\\\"]
rows += [f"\\quad {VN[v]} & {au['source_within_class']['ham'][v]:.3f} & {au['source_within_class']['spam'][v]:.3f} & "
         f"{au['label_from_view_pool'][v]:.3f} \\\\" for v in VN]
rows += ["\\multicolumn{4}{@{}l}{\\emph{Six training-pool sources, model views}} \\\\"]
rows += [f"\\quad {VN[v]} & " + (f"{au10['source_within_class']['ham'][v]:.3f} & {au10['source_within_class']['spam'][v]:.3f} & "
                                   f"{au10['label_from_view_pool'][v]:.3f}" if au10 else "\\TBD & \\TBD & \\TBD") + " \\\\" for v in VN]
(OUT / "tab_audit.tex").write_text("\n".join(rows) + "\n")

# Table: leave-one-source-out on ham
rows, srcs = [], ["apache", "gnu", "python", "fedora"]
lo = [json.load(open(ROOT / f"results/baselines/classical_loso_seed{s}.json")) for s in range(5)]
for key, name in CLASSICAL:
    rows.append(f"{name} & " + " & ".join(ms([d["held_out"][h]["models"][key]["test"]["tpr_fpr_0p1"] for d in lo])
                                         for h in srcs) + " \\\\")
c10 = [json.load(open(f)) for f in sorted(glob.glob(str(A10 / "a10_classical_seed*.json")))]
for ctl, label in (("loso_full_pool", "full pool"), ("loso_size_matched", "size-matched pool")):
    for key, name in CLASSICAL:
        if len(c10) == 5:
            vals = [[d[ctl][h][key] if ctl == "loso_full_pool" else d[ctl][h][key]["tpr_fpr_0p1"] for d in c10] for h in srcs]
            rows.append(f"{name}, {label} & " + " & ".join(ms(v) for v in vals) + " \\\\")
        else:
            rows.append(f"{name}, {label} & " + " & ".join(["\\TBD"] * 4) + " \\\\")
for _, name in TRANSFORMERS:
    rows.append(f"{name} & " + " & ".join(["\\TBD"] * 4) + " \\\\")
rows.append(f"{PROPOSED} & " + " & ".join(["\\TBD"] * 4) + " \\\\")
rows.append("\\midrule\nHam in the test & " + " & ".join(f"{lo[0]['held_out'][h]['n_test_ham']:,}" for h in srcs) + " \\\\")
(OUT / "tab_loso.tex").write_text("\n".join(rows) + "\n")
num["loso_n_ham"] = {h: lo[0]["held_out"][h]["n_test_ham"] for h in srcs}

# Table: data of seed 0, from the split manifest of the merged A7 data that every kernel read; checked against the
# seed-0 score file of the proposed model
K = ROOT / "results/step3_a7/out"
man = {k.split("/")[1]: v for k, v in json.load(open(K / "split_manifest.json"))["counts"].items() if k.startswith("seed0/")}
g = pd.read_parquet(glob.glob(str(ROOT / "results/mustguard/confirm/s0/out/scores_*_best.parquet"))[0])
cnt = g.groupby(["set", "label"]).size().unstack()
for k in ("val", "cal", "s1_test", "s2_test", "s3_test"):
    assert (cnt.loc[k, 0], cnt.loc[k, 1]) == (man[k]["ham"], man[k]["spam"]), k
rf = json.load(open(ROOT / "results/rewrite_filter.json"))
n_para = json.load(open(sorted(glob.glob(str(ROOT / "results/mustguard/confirm/s0/out/log_*_c00.json")))[0]))["n_para"]
f = lambda x: f"{x:,}"
rows = [f"Training & 2015--2020 & {f(man['train']['ham'])} & {f(man['train']['spam'])} \\\\",
        f"$G_1$ paraphrases & 2015--2020 & -- & {f(n_para)} \\\\",
        f"Validation & 2015--2020 & {f(man['val']['ham'])} & {f(man['val']['spam'])} \\\\",
        f"Calibration & 2015--2020 & {f(man['cal']['ham'])} & {f(man['cal']['spam'])} \\\\",
        f"In-distribution test & 2015--2020 & {f(man['s1_test']['ham'])} & {f(man['s1_test']['spam'])} \\\\",
        f"Cross-corpus test & 2002, 2006 & {f(man['s2_test']['ham'])} & {f(man['s2_test']['spam'])} \\\\",
        f"Temporal test & 2021--2026 & {f(man['s3_test']['ham'])} & {f(man['s3_test']['spam'])} \\\\",
        f"$G_2$ stress test & 2021--2026 & {f(int(cnt.loc['g2_rewrite', 0]))} & {f(int(cnt.loc['g2_rewrite', 1]))} \\\\"]
(OUT / "tab_data.tex").write_text("\n".join(rows) + "\n")
bs = json.load(open(K / "build_stats.json"))["by_source_label"]
NAMES = [("apache", "Apache lists", "2015--2026"), ("gnu", "GNU lists", "2015--2026"), ("python", "Python lists", "2015--2026"),
         ("fedora", "Fedora lists", "2015--2026"), ("untroubled", "Untroubled spam archive", "2016--2026"),
         ("nazario", "Nazario phishing corpus", "2015--2025"), ("spamassassin", "SpamAssassin public corpus", "2002"),
         ("enron_spam", "Enron-Spam", "2006")]
rows = [f"{n} & {y} & {f(bs.get(k + '|0', 0)) if bs.get(k + '|0') else '--'} & {f(bs.get(k + '|1', 0)) if bs.get(k + '|1') else '--'} \\\\"
        for k, n, y in NAMES]
(OUT / "tab_corpora.tex").write_text("\n".join(rows) + "\n")
num["data"] = {**man, "n_para_seed0": n_para, "rewrite_filter": rf, "by_source_label": bs}
(ROOT / "results/manuscript_numbers.json").write_text(json.dumps(num, indent=1, default=float))
print("tables:", sorted(p.name for p in OUT.glob("*.tex")))
print(cnt)


# A10 tables from results/extra_analysis.json (saved scores) and the classical JSON files
ex = json.load(open(ROOT / "results/extra_analysis.json"))
cl = [json.load(open(ROOT / f"results/baselines/classical_seed{s}.json"))["models"] for s in range(5)]
S4 = [("s1_test", "In-distribution"), ("s3_test", "Temporal"), ("s2_test", "Cross-corpus"), ("g2_rewrite", "$G_2$ rewrites")]
pp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
rows = []
for s, sname in S4:
    rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{sname}}}}} \\\\")
    done = {r["baseline"] for r in ex["paired_tests"][s]}
    for key, name in CLASSICAL:
        if name not in done:
            rows.append(f"\\quad {name} & 5 & \\TBD & \\TBD & \\TBD & \\TBD \\\\")
    for r in ex["paired_tests"][s]:
        rows.append(f"\\quad {r['baseline']} & {len(r['seeds'])} & {100 * r['diff']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] & "
                    f"{pp(r['p_boot_holm'])} & {r['mcnemar_b']:,}/{r['mcnemar_c']:,} & {pp(r['p_mcnemar_holm'])} \\\\")
(OUT / "tab_stats.tex").write_text("\n".join(rows) + "\n")

rows = []
for key, name in CLASSICAL:
    rows.append(f"{name} & " + " & ".join(ms([c[key]["test"][s]["tpr_fpr_1"] for c in cl]) for s in ("s1", "s3", "s2")) + " & "
                + " & ".join(f"{np.mean([c[key]['test'][s]['pr_auc'] for c in cl]):.3f}" for s in ("s1", "s3", "s2")) + " \\\\")
for _, name in TRANSFORMERS + [(None, PROPOSED)]:
    if name not in ex["secondary"]:
        rows.append(f"{name} & " + " & ".join(["\\TBD"] * 6) + " \\\\"); continue
    sc = ex["secondary"][name]
    rows.append(f"{nm(name)} & " + " & ".join(ms(sc[s]["tpr_fpr_1"]) for s in ("s1_test", "s3_test", "s2_test")) + " & "
                + " & ".join(f"{np.mean(sc[s]['pr_auc']):.3f}" for s in ("s1_test", "s3_test", "s2_test")) + " \\\\")
(OUT / "tab_secondary.tex").write_text("\n".join(rows) + "\n")

years = sorted(int(y) for y in next(iter(ex["s3_per_year"].values())))
mn = lambda v: f"{100 * np.mean(v):.1f}"
rows = []
for key, name in CLASSICAL:
    rows.append(f"{name} & " + " & ".join(mn([c[key]["test"][f"s3_{y}"]["tpr_fpr_0p1"] for c in cl]) for y in years) + " \\\\")
for _, name in TRANSFORMERS + [(None, PROPOSED)]:
    if name not in ex["s3_per_year"]:
        rows.append(f"{name} & " + " & ".join(["\\TBD"] * len(years)) + " \\\\"); continue
    rows.append(f"{nm(name)} & " + " & ".join(mn([v for v in ex["s3_per_year"][name][str(y)] if v is not None]) for y in years) + " \\\\")
yc = ex["s3_year_counts"]
rows.append("\\midrule\nHam & " + " & ".join(f"{int(yc['0'][f'{y}.0']):,}" for y in years) + " \\\\")
rows.append("Spam & " + " & ".join(f"{int(yc['1'][f'{y}.0']):,}" for y in years) + " \\\\")
(OUT / "tab_year.tex").write_text("\n".join(rows) + "\n")
(OUT / "tab_year_head.tex").write_text(" & ".join(str(y) for y in years) + "\n")

rows = []
for key, name in CLASSICAL:
    g = lambda src, m: ms([c[key]["test"][f"s2_{src}"][m] for c in cl])
    rows.append(f"{name} & {g('spamassassin', 'tpr_fpr_0p1')} & {g('enron_spam', 'tpr_fpr_0p1')} & {g('spamassassin', 'cal_thr_fpr')} & {g('enron_spam', 'cal_thr_fpr')} \\\\")
for _, name in TRANSFORMERS + [(None, PROPOSED)]:
    if name not in ex["s2_per_corpus"]:
        rows.append(f"{name} & \\TBD & \\TBD & \\TBD & \\TBD \\\\"); continue
    d = ex["s2_per_corpus"][name]
    rows.append(f"{nm(name)} & {ms(d['spamassassin']['tpr'])} & {ms(d['enron_spam']['tpr'])} & {ms(d['spamassassin']['realized_fpr'])} & {ms(d['enron_spam']['realized_fpr'])} \\\\")
(OUT / "tab_s2split.tex").write_text("\n".join(rows) + "\n")

rows = []
for _, name in CLASSICAL + TRANSFORMERS + [(None, PROPOSED)]:
    if name not in ex["per_class"]:
        rows.append(f"{name} & \\TBD & \\TBD & \\TBD & \\TBD \\\\"); continue
    d = ex["per_class"][name]
    rows.append(f"{nm(name)} & " + " & ".join(ms(d[s][c]) for s in ("s3_test", "g2_rewrite") for c in ("spam", "phishing")) + " \\\\")
(OUT / "tab_class.tex").write_text("\n".join(rows) + "\n")

rows = []
for _, name in TRANSFORMERS + [(None, PROPOSED)]:
    if name not in ex["g2_flips"]:
        rows.append(f"{name} & \\TBD & \\TBD & \\TBD & \\TBD \\\\"); continue
    f = ex["g2_flips"][name]; lost, gained = sum(r["lost"] for r in f), sum(r["gained"] for r in f)
    from scipy.stats import binomtest
    p = binomtest(lost, lost + gained, 0.5).pvalue if lost + gained else 1.0
    rows.append(f"{nm(name)} & {len(f)} & {lost} & {gained} & {pp(p)} \\\\")
(OUT / "tab_flips.tex").write_text("\n".join(rows) + "\n")
print("A10 tables written")

# A7 deviation: the single-ham-source condition of A1 (Apache ham only), kept and reported as A7 promised
rows = []
a1 = [json.load(open(ROOT / f"results/baselines_A1/classical_seed{s}.json"))["models"] for s in range(5)]
for key, name in CLASSICAL:
    rows.append(f"{name} & 5 & " + " & ".join(ms([c[key]["test"][s]["tpr_fpr_0p1"] for c in a1]) for s in ("s1", "s3", "s2"))
                + f" & {ms([c[key]['test']['s2']['cal_thr_fpr'] for c in a1])} \\\\")
for key, name in TRANSFORMERS:
    rs = score_runs(glob.glob(str(ROOT / f"results/gpu/tb1-{key}-s*/out/scores_*_best.parquet")))
    if rs:
        rows.append(f"{name} & {len(rs)} & " + " & ".join(ms([r[s]["tpr"] for r in rs]) for s in ("s1", "s3", "s2"))
                    + f" & {ms([r['s2']['fpr'] for r in rs])} \\\\")
(OUT / "tab_a1.tex").write_text("\n".join(rows) + "\n")
print("A1 table written")
