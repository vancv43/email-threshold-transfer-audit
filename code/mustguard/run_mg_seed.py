"""Batch-job entry point for MuST-Guard: tune one encoder on one seed with the 20-configuration grid of A6, then score
the test sets once with the configuration that has the best validation PR-AUC (the baselines' rule).

Two workers run in parallel, one per T4 GPU; each keeps only the checkpoint of its best configuration.
--pilot: train on 10% of the training split, no scoring of any test set; writes pilot_<encoder>.json.
Extra arguments after the seed are passed to train_mg.py (ablations and curricula, e.g. --use body --para 0).
A11: configurations are compared by the selection key of train_mg.py; --grid compact uses the 8 of T5.
Usage inside the kernel: run_mg_seed.py <hf_model> <seed> [--pilot] [--name NAME] [train_mg.py args]
"""
import argparse, glob, itertools, json, os, pathlib, subprocess, sys, threading

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("seed", type=int)
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--name", default="mg")
ap.add_argument("--grid", default="full", choices=["full", "compact"])  # A11 T5: compact = 8 configurations
ap.add_argument("--fixed", default="")  # A13: "lr,batch,wd" of the configuration already selected
a, extra = ap.parse_known_args()
MODEL, SEED = a.model, a.seed
DATA = os.environ.get("MG_DATA") or glob.glob("input/**/mg.parquet", recursive=True)[0].rsplit("/", 1)[0]
CODE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path("work/out"); OUT.mkdir(parents=True, exist_ok=True)
SHORT = f"{a.name}-{MODEL.rstrip('/').split('/')[-1]}" + ("-pilot" if a.pilot else "")
GRID = [dict(lr=lr, batch=b, wd=wd) for lr, b, wd in
        itertools.product([1e-5, 2e-5, 3e-5, 4e-5, 5e-5], [16, 32], [0.0, 0.01])]  # same 20 as the baselines
if a.grid == "compact":
    GRID = [dict(lr=lr, batch=16, wd=wd) for lr, wd in itertools.product([1e-5, 2e-5, 3e-5, 5e-5], [0.0, 0.01])]
if a.fixed:
    lr_, b_, wd_ = a.fixed.split(","); GRID = [dict(lr=float(lr_), batch=int(b_), wd=float(wd_))]
PILOT = ["--pilot_frac", "0.1"] if a.pilot else []


def tag(i):
    return f"{SHORT}_s{SEED}_c{i:02d}"


def run(i, gpu):
    c = GRID[i]
    cmd = [sys.executable, str(CODE / "train_mg.py"), "--model", MODEL, "--seed", str(SEED), "--lr", str(c["lr"]),
           "--batch", str(c["batch"]), "--wd", str(c["wd"]), "--data", DATA, "--out", str(OUT), "--tag", tag(i),
           "--ckpt", f"/tmp/{tag(i)}.pt"] + PILOT + extra
    subprocess.run(cmd, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    log = json.loads((OUT / f"log_{tag(i)}.json").read_text())
    return tuple(log.get("sel_key", [log["val_pr_auc"]]))  # A11 logs carry the selection key, v1 logs PR-AUC


best = {}  # worker -> (val, index)


def worker(gpu, idx):
    for i in idx:
        v = run(i, gpu)
        prev = best.get(gpu)
        if prev is None or v > prev[0]:
            if prev:
                os.remove(f"/tmp/{tag(prev[1])}.pt")
            best[gpu] = (v, i)
        else:
            os.remove(f"/tmp/{tag(i)}.pt")


ts = [threading.Thread(target=worker, args=(g, list(range(g, len(GRID), 2)))) for g in (0, 1)]
[t.start() for t in ts]; [t.join() for t in ts]
assert len(best) == min(2, len(GRID)), "a worker failed"
v, i = max(best.values(), key=lambda t: (t[0], -t[1]))  # ties go to the earlier grid entry
logs = [json.loads((OUT / f"log_{tag(j)}.json").read_text()) for j in range(len(GRID))]
sel = {"model": MODEL, "seed": SEED, "name": a.name, "extra": extra, "grid": a.grid, "index": i, "config": GRID[i],
       "sel_key": list(v), "val_pr_auc": logs[i]["val_pr_auc"], "val_tpr_fpr_0p1": logs[i]["val_tpr_fpr_0p1"]}
if a.pilot:
    sel["grid_best_val_tpr_fpr_0p1"] = max(l["val_tpr_fpr_0p1"] for l in logs)
    (OUT / f"pilot_{SHORT}.json").write_text(json.dumps(sel, indent=1))
else:
    subprocess.run([sys.executable, str(CODE / "train_mg.py"), "--model", MODEL, "--seed", str(SEED), "--data", DATA,
                    "--out", str(OUT), "--tag", f"{SHORT}_s{SEED}_best", "--ckpt", f"/tmp/{tag(i)}.pt", "--infer"]
                   + extra, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    (OUT / f"selected_{SHORT}_s{SEED}.json").write_text(json.dumps(sel, indent=1))
open(OUT / f"DONE_{SHORT}_s{SEED}", "w").write("ok")
