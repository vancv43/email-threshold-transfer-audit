"""Batch-job entry point: tune one transformer baseline on one seed with the 20-configuration grid of A6, then score
the test sets once with the configuration that has the best validation PR-AUC.

Two workers run in parallel, one per T4 GPU. Each worker keeps only the checkpoint of its best configuration.
Usage inside the kernel: run_model_seed.py <hf_model> <seed>; writes work/out/
"""
import argparse, glob, itertools, json, os, pathlib, subprocess, sys, threading

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("seed", type=int)
ap.add_argument("--select", default="pr", choices=["pr", "ood"])  # A11 T1
ap.add_argument("--grid", default="full", choices=["full", "compact"])  # A11 T5
ap.add_argument("--name", default="")
ap.add_argument("--fixed", default="")  # A13: "lr,batch,wd" of the configuration already selected
A = ap.parse_args()
MODEL, SEED = A.model, A.seed
DATA = os.environ.get("TB_DATA") or glob.glob("input/**/text.parquet", recursive=True)[0].rsplit("/", 1)[0]
CODE = pathlib.Path(__file__).resolve().parent if os.environ.get("TB_DATA") else pathlib.Path(DATA)
OUT = pathlib.Path("work/out"); OUT.mkdir(parents=True, exist_ok=True)
SHORT = (A.name + "-" if A.name else "") + MODEL.split("/")[-1]
GRID = [dict(lr=lr, batch=b, wd=wd) for lr, b, wd in
        itertools.product([1e-5, 2e-5, 3e-5, 4e-5, 5e-5], [16, 32], [0.0, 0.01])]  # 20 configurations
if A.grid == "compact":
    GRID = [dict(lr=lr, batch=16, wd=wd) for lr, wd in itertools.product([1e-5, 2e-5, 3e-5, 5e-5], [0.0, 0.01])]
if A.fixed:
    lr_, b_, wd_ = A.fixed.split(","); GRID = [dict(lr=float(lr_), batch=int(b_), wd=float(wd_))]
MAX_LEN = 300 if MODEL.split("/")[-1] == "bert-base-cased" else 256  # RTIS used 300 tokens


def tag(i):
    return f"{SHORT}_s{SEED}_c{i:02d}"


def run(i, gpu):
    c = GRID[i]
    cmd = [sys.executable, f"{CODE}/ft.py", "--select", A.select, "--model", MODEL, "--seed", str(SEED), "--lr", str(c["lr"]),
           "--batch", str(c["batch"]), "--wd", str(c["wd"]), "--epochs", "3", "--max_len", str(MAX_LEN),
           "--data", DATA, "--out", str(OUT), "--tag", tag(i), "--ckpt", f"/tmp/{tag(i)}.pt"]
    subprocess.run(cmd, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    log = json.loads((OUT / f"log_{tag(i)}.json").read_text())
    return tuple(log.get("sel_key", [log["val_pr_auc"]]))


best = {}  # worker -> (val, index)


def worker(gpu, idx):
    for i in idx:
        if (OUT / f"log_{tag(i)}.json").exists():
            continue
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
v, i = max(best.values(), key=lambda t: (t[0], -t[1]))  # ties go to the earlier grid entry
c = GRID[i]
subprocess.run([sys.executable, f"{CODE}/ft.py", "--model", MODEL, "--seed", str(SEED), "--max_len", str(MAX_LEN),
                "--data", DATA, "--out", str(OUT), "--tag", f"{SHORT}_s{SEED}_best", "--ckpt", f"/tmp/{tag(i)}.pt",
                "--infer"], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
(OUT / f"selected_{SHORT}_s{SEED}.json").write_text(json.dumps({"model": MODEL, "seed": SEED, "index": i,
                                                                 "config": c, "sel_key": list(v), "select": A.select,
                                                                 "grid": A.grid, "max_len": MAX_LEN}))
open(OUT / "DONE", "w").write("ok")
