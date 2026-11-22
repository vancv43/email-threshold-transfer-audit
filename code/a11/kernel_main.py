"""A11 Batch-job entry point. Jobs:
  mg2 <seed>      MuST-Guard v2 (T1-T5) on one seed, GPU
  distil <seed>   DistilBERT with the T1 rule and the compact grid on one seed, GPU
  classical       the three classical models with the T1 rule, seeds 0-4, CPU
  smoke           every job above on a small sample with a tiny encoder; checks the outputs, GPU
Inputs: dataset a11-data (messages.parquet, splits.parquet, rewrite_g1_kept.parquet, rewrite_g2_kept.parquet)
and dataset a11-code (code/). Output: work/out/ and DONE.
"""
import glob, json, os, pathlib, shutil, subprocess, sys
import pandas as pd

JOB, SEED = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else None)
W, OUT = pathlib.Path("work/proj"), pathlib.Path("work/out")
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def find(name):
    return glob.glob(f"input/**/{name}", recursive=True)[0]


if not W.exists():
    shutil.copytree(pathlib.Path(find("t1_split.py")).parents[1], W / "code")
    for sub, name in [("processed", "messages.parquet"), ("processed", "splits.parquet"),
                      ("rewrites/g1", "rewrite_g1_kept.parquet"), ("rewrites/g2", "rewrite_g2_kept.parquet")]:
        (W / "data" / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy(find(name), W / "data" / sub / name)
C, D = W / "code", W / "data"
MSG, SPL = D / "processed/messages.parquet", D / "processed/splits.parquet"
G1, G2 = D / "rewrites/g1/rewrite_g1_kept.parquet", D / "rewrites/g2/rewrite_g2_kept.parquet"
sys.path.insert(0, str(C / "a11"))
import t1_split  # noqa: E402

TINY = "hf-internal-testing/tiny-random-bert"
MG2 = ["--grid", "compact", "--select", "ood", "--aux", "0.5", "--ls", "0.1", "--mdrop", "0.3"]


def run(cmd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env={**os.environ, **(env or {})})


def t1(path):
    sp, msg = pd.read_parquet(path), pd.read_parquet(MSG, columns=["msg_id", "label", "source"])
    out, stats = t1_split.apply(sp, msg)
    out.to_parquet(path, index=False)
    stats.to_json(OUT / "t1_split_stats.json", orient="records")


def subsample(data_dir, table, n=240, seed=0):
    """smoke only: keep seed 0 and n rows per split and label; shrink the G2 sets."""
    sp = pd.read_parquet(data_dir / "splits.parquet")
    lab = pd.read_parquet(data_dir / table, columns=["msg_id", "label"])
    sp = sp[sp.seed == seed].merge(lab, on="msg_id")
    sp = sp.groupby(["split", "label"], group_keys=False).apply(lambda g: g.sample(min(len(g), n // 2), random_state=0))
    sp[["msg_id", "seed", "split"]].to_parquet(data_dir / "splits.parquet", index=False)
    g2 = pd.read_parquet(data_dir / "g2_sets.parquet")
    g2.groupby(["set", "label"], group_keys=False).apply(lambda g: g.head(60)).to_parquet(data_dir / "g2_sets.parquet", index=False)


def mg2(seed, model="microsoft/deberta-v3-small", smoke=False):
    data = pathlib.Path("work/mg_data")
    run([PY, C / "mustguard/make_mg_data.py", MSG, SPL, G1, G2, data])
    t1(data / "splits.parquet")
    if smoke:
        subsample(data, "mg.parquet")
    run([PY, C / "mustguard/run_mg_seed.py", model, seed, "--name", "mg2"] + MG2, {"MG_DATA": str(data)})


def distil(seed, model="distilbert-base-uncased", smoke=False):
    data = pathlib.Path("work/tb_data")
    run([PY, C / "gpu/make_gpu_data.py", data])
    t1(data / "splits.parquet")
    if smoke:
        subsample(data, "text.parquet")
    run([PY, C / "gpu/run_model_seed.py", model, seed, "--select", "ood", "--grid", "compact", "--name", "t1"],
        {"TB_DATA": str(data)})


def classical(smoke=False):
    t1(SPL)
    run([PY, C / "a11/classical_t1.py", OUT / "classical"] + (["--seeds", "0", "--sub", "400"] if smoke else []))


if JOB == "mg2":
    mg2(SEED)
elif JOB == "distil":
    distil(SEED)
elif JOB == "classical":
    classical()
elif JOB == "smoke":
    mg2(0, TINY, smoke=True)
    distil(0, TINY, smoke=True)
    classical(smoke=True)
    chk = {}
    sel = json.loads(next(OUT.glob("selected_mg2-*_s0.json")).read_text())
    chk["mg2_sel_key_has_ood"] = len(sel["sel_key"]) == 2 and sel["extra"][-8:] == MG2[2:]
    chk["mg2_grid_8"] = len(list(OUT.glob("log_mg2-*_s0_c*.json"))) == 8
    sc = pd.read_parquet(next(OUT.glob("scores_mg2-*_s0_best.parquet")))
    chk["mg2_scores_ood_ham"] = "ood_ham" in set(sc["set"])
    tb = json.loads(next(OUT.glob("selected_t1-*_s0.json")).read_text())
    chk["distil_sel_key_has_ood"] = len(tb["sel_key"]) == 2 and tb["select"] == "ood"
    cl = json.loads((OUT / "classical/classical_t1_seed0.json").read_text())
    chk["classical_holdout_fedora"] = cl["holdout"] == "fedora" and all(len(v["sel_key"]) == 2 for v in cl["selected"].values())
    lg = json.loads(next(OUT.glob("log_mg2-*_s0_c00.json")).read_text())
    chk["mg2_args"] = lg["aux"] == 0.5 and lg["ls"] == 0.1 and lg["mdrop"] == 0.3 and lg["select"] == "ood"
    (OUT / "smoke_checks.json").write_text(json.dumps(chk, indent=1))
    print("SMOKE", chk, flush=True)
    assert all(chk.values()), chk
for d in (W, pathlib.Path("work/mg_data"), pathlib.Path("work/tb_data")):
    shutil.rmtree(d, ignore_errors=True)  # keep the downloaded output small: inputs are not copied back
open(OUT / "DONE", "w").write("ok")
