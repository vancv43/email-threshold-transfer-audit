"""Mean and standard deviation over seeds 0-4 of the classical baselines. Output: results/baselines/classical_summary.json"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
runs = [json.loads((ROOT / f"results/baselines/classical_seed{s}.json").read_text()) for s in range(5)]
out = {}
for m in runs[0]["models"]:
    out[m] = {"selected": [r["models"][m]["selected"] for r in runs], "test": {}}
    for t in runs[0]["models"][m]["test"]:
        out[m]["test"][t] = {}
        for k in ["tpr_fpr_0p1", "tpr_fpr_1", "pr_auc", "roc_auc", "f1", "ece", "cal_thr_tpr", "cal_thr_fpr"]:
            v = [r["models"][m]["test"][t].get(k) for r in runs]
            v = [x for x in v if x is not None]
            if v:
                out[m]["test"][t][k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
(ROOT / "results/baselines/classical_summary.json").write_text(json.dumps(out, indent=1))
for m, d in out.items():
    print(m, [f"{s['view']}/{s['param']:.3g}" for s in d["selected"]])
    for t in ["s1", "s2", "s2_spamassassin", "s2_enron_spam", "s3", "s3_2021", "s3_2026"]:
        x = d["test"][t]
        print(f"  {t:16s} TPR@0.1% {x['tpr_fpr_0p1']['mean']:.4f}±{x['tpr_fpr_0p1']['sd']:.4f}  PR-AUC {x['pr_auc']['mean']:.4f}"
              f"  F1 {x['f1']['mean']:.4f}  ECE {x['ece']['mean']:.4f}  calthr TPR {x['cal_thr_tpr']['mean']:.3f} FPR {x['cal_thr_fpr']['mean']:.4f}")
