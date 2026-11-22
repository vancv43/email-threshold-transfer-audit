"""Collect the configuration each model selected on each seed (A6/A8 runs) into a13/configs.json."""
import glob, json, pathlib, re
R = pathlib.Path(__file__).resolve().parents[2] / "results"
cfg = {"mustguard": {}, "classical": {}}
for f in glob.glob(str(R / "mustguard/confirm/s*/out/selected_mg-*.json")):
    j = json.load(open(f)); cfg["mustguard"][str(j["seed"])] = j["config"]
for m in ["bert-base-cased", "distilbert-base-uncased", "roberta-base", "deberta-v3-base"]:
    key = "microsoft/deberta-v3-base" if m == "deberta-v3-base" else m
    cfg[key] = {}
    for f in glob.glob(str(R / f"gpu7/tb-{m}-s*/out/selected_*.json")):
        j = json.load(open(f)); cfg[key][str(j["seed"])] = j["config"]
for s in range(5):
    d = json.load(open(R / f"baselines/classical_seed{s}.json"))["models"]
    cfg["classical"][str(s)] = {k: {"view": v["selected"]["view"], "param": v["selected"]["param"]} for k, v in d.items()}
(pathlib.Path(__file__).parent / "configs.json").write_text(json.dumps(cfg, indent=1))
print({k: sorted(v) for k, v in cfg.items()})
