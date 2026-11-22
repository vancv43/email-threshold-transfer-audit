"""A13 Batch-job entry point. Jobs:
  mg                 MuST-Guard (v1 settings), seeds 0-4, each retrained once at its selected configuration, GPU
  tb <hf_model>      one transformer baseline, seeds 0-4, the same way, GPU
  classical          the three classical baselines, seeds 0-4, CPU
  smoke              mg, tb and classical on a small sample with a tiny encoder; checks the outputs
Every job first applies a13_split.py (adaptation ham into train, tcal, evaluation part of S2).
Inputs: dataset jnsm-a13-data (messages, splits, G1, G2) and dataset jnsm-a13-code (code/, with a13/configs.json).
Output: work/out/ and DONE.
"""
import glob, json, os, pathlib, shutil, subprocess, sys
import pandas as pd

JOB, ARG = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
W, OUT = pathlib.Path("work/proj"), pathlib.Path("work/out")
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def find(name):
    return glob.glob(f"input/**/{name}", recursive=True)[0]


if not W.exists():
    shutil.copytree(pathlib.Path(find("a13_split.py")).parents[1], W / "code")
    for sub, name in [("processed", "messages.parquet"), ("processed", "splits.parquet"),
                      ("rewrites/g1", "rewrite_g1_kept.parquet"), ("rewrites/g2", "rewrite_g2_kept.parquet")]:
        (W / "data" / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy(find(name), W / "data" / sub / name)
C, D = W / "code", W / "data"
MSG, SPL = D / "processed/messages.parquet", D / "processed/splits.parquet"
G1, G2 = D / "rewrites/g1/rewrite_g1_kept.parquet", D / "rewrites/g2/rewrite_g2_kept.parquet"
CFG = json.load(open(C / "a13/configs.json"))
sys.path.insert(0, str(C / "a13"))
import a13_split  # noqa: E402

TINY = "hf-internal-testing/tiny-random-bert"


def run(cmd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env={**os.environ, **(env or {})})


def split13(path):
    sp, msg = pd.read_parquet(path), pd.read_parquet(MSG, columns=["msg_id", "label", "source"])
    out, stats = a13_split.apply(sp, msg)
    out.to_parquet(path, index=False)
    stats.to_json(OUT / "a13_split_stats.json", orient="records")


def subsample(data_dir, table, n=240, seed=0):
    sp = pd.read_parquet(data_dir / "splits.parquet")
    lab = pd.read_parquet(data_dir / table, columns=["msg_id", "label"])
    sp = sp[sp.seed == seed].merge(lab, on="msg_id")
    sp = sp.groupby(["split", "label"], group_keys=False).apply(lambda g: g.sample(min(len(g), n // 2), random_state=0))
    sp[["msg_id", "seed", "split"]].to_parquet(data_dir / "splits.parquet", index=False)
    g2 = pd.read_parquet(data_dir / "g2_sets.parquet")
    g2.groupby(["set", "label"], group_keys=False).apply(lambda g: g.head(60)).to_parquet(data_dir / "g2_sets.parquet", index=False)


def fixed(c):
    return f"{c['lr']},{c['batch']},{c['wd']}"


def mg(seeds, model="microsoft/deberta-v3-small", smoke=False):
    data = pathlib.Path("work/mg_data")
    run([PY, C / "mustguard/make_mg_data.py", MSG, SPL, G1, G2, data])
    split13(data / "splits.parquet")
    if smoke:
        subsample(data, "mg.parquet")
    for s in seeds:
        run([PY, C / "mustguard/run_mg_seed.py", model, s, "--name", "a13", "--fixed", fixed(CFG["mustguard"][str(s)])],
            {"MG_DATA": str(data)})


def tb(name, seeds, model=None, smoke=False, run_name="a13"):
    data = pathlib.Path("work/tb_data")
    run([PY, C / "gpu/make_gpu_data.py", data])
    split13(data / "splits.parquet")
    if smoke:
        subsample(data, "text.parquet")
    for s in seeds:
        run([PY, C / "gpu/run_model_seed.py", model or name, s, "--name", run_name, "--fixed", fixed(CFG[name][str(s)])],
            {"TB_DATA": str(data)})


def classical(smoke=False):
    split13(SPL)
    run([PY, C / "a13/classical_a13.py", OUT / "classical", C / "a13/configs.json"] + (["--seeds", "0", "--sub", "400"] if smoke else []))


if JOB == "mg":
    mg(range(5))
elif JOB == "tb":
    tb(ARG, range(5))
elif JOB == "classical":
    classical()
elif JOB == "smoke":
    mg([0], TINY, smoke=True)
    tb("distilbert-base-uncased", [0], TINY, smoke=True, run_name="a13tb")  # own name: same tiny encoder as mg
    classical(smoke=True)
    chk = {}
    st = pd.DataFrame(json.loads((OUT / "a13_split_stats.json").read_text()))
    chk["tcal_is_ham"] = set(st.loc[st.split == "tcal", "label"]) == {0}
    sc = pd.read_parquet(next(OUT.glob("scores_a13-*_s0_best.parquet")))
    chk["scores_have_tcal"] = "tcal" in set(sc["set"])
    chk["mg_one_config"] = len(list(OUT.glob("log_a13-tiny-random-bert_s0_c*.json"))) >= 1
    cl = pd.read_parquet(OUT / "classical/classical_a13_scores_seed0.parquet")
    chk["classical_sets"] = {"tcal", "s1_test", "s2_test", "s3_test"} <= set(cl["set"])
    (OUT / "smoke_checks.json").write_text(json.dumps(chk, indent=1))
    print("SMOKE", chk, flush=True)
    assert all(chk.values()), chk
for d in (W, pathlib.Path("work/mg_data"), pathlib.Path("work/tb_data")):
    shutil.rmtree(d, ignore_errors=True)
open(OUT / "DONE", "w").write("ok")
