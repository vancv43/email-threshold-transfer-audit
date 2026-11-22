"""Mean and standard deviation over seeds 0-4 of the G2 stress test (A5). Output: results/baselines/classical_g2_summary.json"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
runs = [json.loads((ROOT / f"results/baselines/classical_g2_seed{s}.json").read_text()) for s in range(5)]
out = {}
for m in runs[0]["models"]:
    g = lambda f: [f(r["models"][m]) for r in runs]
    vals = {"original_tpr_fpr_0p1": g(lambda d: d["g2_original"]["tpr_fpr_0p1"]),
            "rewrite_tpr_fpr_0p1": g(lambda d: d["g2_rewrite"]["tpr_fpr_0p1"]),
            "drop_tpr_fpr_0p1": g(lambda d: d["drop_tpr_fpr_0p1"]),
            "original_cal_thr_tpr": g(lambda d: d["g2_original"]["cal_thr_tpr"]),
            "rewrite_cal_thr_tpr": g(lambda d: d["g2_rewrite"]["cal_thr_tpr"]),
            "refit_matches": g(lambda d: abs(d["s3_tpr_fpr_0p1_refit"] - d["s3_tpr_fpr_0p1_stored"]) < 1e-9)}
    out[m] = {k: ({"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} if k != "refit_matches" else all(v))
              for k, v in vals.items()}
    print(m, {k: (round(v["mean"], 4), round(v["sd"], 4)) if isinstance(v, dict) else v for k, v in out[m].items()})
(ROOT / "results/baselines/classical_g2_summary.json").write_text(json.dumps(out, indent=1))
