"""A14 Batch-job entry point (PROTOCOL.md A14). Jobs:
  abl       A14a: four ablation variants of MuST-Guard, seeds 0-2, each at the seed's selected configuration
  frozen    A14b: stage 1 without target ham, then one epoch with the adaptation ham, gate frozen and trainable, seeds 0-4
  smoke     both jobs on a small sample with a tiny encoder; checks the outputs
Two queues run in parallel, one per T4 GPU. Inputs: dataset a13-data and the code dataset (code/, a13/configs.json).
Output: work/out/ and DONE.
"""
import glob, json, os, pathlib, shutil, subprocess, sys, threading
import pandas as pd

JOB = sys.argv[1]
W, OUT = pathlib.Path("work/proj"), pathlib.Path("work/out")
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def find(name):
    return glob.glob(f"input/**/{name}", recursive=True)[0]


if not W.exists():
    shutil.copytree(pathlib.Path(find("kernel_main_a14.py")).parents[1], W / "code")
    for sub, name in [("processed", "messages.parquet"), ("processed", "splits.parquet"),
                      ("rewrites/g1", "rewrite_g1_kept.parquet"), ("rewrites/g2", "rewrite_g2_kept.parquet")]:
        (W / "data" / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy(find(name), W / "data" / sub / name)
C, D = W / "code", W / "data"
MSG, SPL = D / "processed/messages.parquet", D / "processed/splits.parquet"
G1, G2 = D / "rewrites/g1/rewrite_g1_kept.parquet", D / "rewrites/g2/rewrite_g2_kept.parquet"
CFG = json.load(open(C / "a13/configs.json"))["mustguard"]
sys.path.insert(0, str(C / "a13"))
import a13_split  # noqa: E402

SMOKE = JOB == "smoke"
MODEL = "hf-internal-testing/tiny-random-bert" if SMOKE else "microsoft/deberta-v3-small"
TRAIN = C / "mustguard/train_mg.py"
VARIANTS = {"noaug": ["--para", "0", "--obf", "0"], "nostruct": ["--use", "body,header,url"],
            "body": ["--use", "body"], "bodynoaug": ["--use", "body", "--para", "0", "--obf", "0"]}
errors = []


def run(cmd, gpu):
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    if r.returncode:
        errors.append(" ".join(map(str, cmd[2:6])))
        raise RuntimeError(cmd)


def make_data(path, a13):
    run([PY, C / "mustguard/make_mg_data.py", MSG, SPL, G1, G2, path], 0)
    if a13:
        sp, msg = pd.read_parquet(path / "splits.parquet"), pd.read_parquet(MSG, columns=["msg_id", "label", "source"])
        out, stats = a13_split.apply(sp, msg)
        out.to_parquet(path / "splits.parquet", index=False)
        stats.to_json(OUT / "a13_split_stats.json", orient="records")
    if SMOKE:
        sp = pd.read_parquet(path / "splits.parquet"); lab = pd.read_parquet(path / "mg.parquet", columns=["msg_id", "label", "source"])
        sp = sp[sp.seed.isin([0])].merge(lab, on="msg_id")
        keep = sp.groupby(["split", "label"], group_keys=False).apply(lambda g: g.sample(min(len(g), 80), random_state=0))
        if a13:  # keep some adaptation ham in train
            ad = sp[(sp.split == "train") & sp.source.isin(["spamassassin", "enron_spam"])].head(40)
            keep = pd.concat([keep, ad]).drop_duplicates("msg_id")
        keep[["msg_id", "seed", "split"]].to_parquet(path / "splits.parquet", index=False)
        g2 = pd.read_parquet(path / "g2_sets.parquet")
        g2.groupby(["set", "label"], group_keys=False).apply(lambda g: g.head(40)).to_parquet(path / "g2_sets.parquet", index=False)


def cfg_args(seed):
    c = CFG[str(seed)]
    return ["--lr", c["lr"], "--batch", c["batch"], "--wd", c["wd"]]


def train_infer(data, tag, seed, gpu, extra, init=None, epochs=3):
    ck = f"/tmp/{tag}.pt"
    base = [PY, TRAIN, "--model", MODEL, "--seed", seed, "--data", data, "--out", OUT, "--ckpt", ck]
    run(base + ["--tag", tag, "--epochs", 1 if SMOKE else epochs] + cfg_args(seed) + extra + (["--init", init] if init else []), gpu)
    run(base + ["--tag", f"{tag}_best", "--infer"] + _arch(extra), gpu)
    return ck


def _arch(extra):  # inference needs only the view set
    return ["--use", extra[extra.index("--use") + 1]] if "--use" in extra else []


def parallel(tasks):
    """tasks: list of callables(gpu); two workers, one per GPU."""
    q = list(tasks)
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not q:
                    return
                t = q.pop(0)
            try:
                t(gpu)
            except Exception as e:  # keep the other queue running; failures are listed in errors
                print("FAILED", e, flush=True)
    ts = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    [t.start() for t in ts]; [t.join() for t in ts]


def abl(seeds):
    data = pathlib.Path("work/mg_data"); make_data(data, a13=False)
    tasks = [lambda gpu, v=v, s=s: train_infer(data, f"a14a-{v}_s{s}", s, gpu, VARIANTS[v]) for s in seeds for v in VARIANTS]
    parallel(tasks)


def frozen(seeds):
    data = pathlib.Path("work/mg13_data"); make_data(data, a13=True)
    for s in seeds:
        ck1 = train_infer(data, f"a14b-stage1_s{s}", s, 0, ["--drop_target_ham", "1"])
        parallel([lambda gpu, fz=fz: train_infer(data, f"a14b-{'frozen' if fz else 'trainable'}_s{s}", s, gpu,
                                                 ["--freeze_gate", str(fz)], init=ck1, epochs=1) for fz in (1, 0)])
        os.remove(ck1)


if JOB == "abl":
    abl([0, 1, 2])
elif JOB == "frozen":
    frozen([0, 1, 2, 3, 4])
elif JOB == "smoke":
    abl([0]); frozen([0])
    chk = {"no_errors": not errors}
    chk["abl_scores"] = len(list(OUT.glob("scores_a14a-*_s0_best.parquet"))) == 4
    chk["frozen_scores"] = len(list(OUT.glob("scores_a14b-*_s0_best.parquet"))) == 3
    lf = json.loads((OUT / "log_a14b-frozen_s0.json").read_text()); ls1 = json.loads((OUT / "log_a14b-stage1_s0.json").read_text())
    chk["stage1_no_target_ham"] = ls1["n_target_ham"] == 0
    chk["stage2_has_target_ham"] = lf["n_target_ham"] > 0 and lf["freeze_gate"] == 1
    sc = pd.read_parquet(OUT / "scores_a14b-frozen_s0_best.parquet"); chk["tcal_scored"] = "tcal" in set(sc["set"])
    sb = pd.read_parquet(OUT / "scores_a14a-body_s0_best.parquet"); chk["body_only_gate"] = bool((sb["gate_body"] > 0.999).all())
    (OUT / "smoke_checks.json").write_text(json.dumps(chk, indent=1)); print("SMOKE", chk, flush=True)
    assert all(chk.values()), chk
(OUT / "errors.json").write_text(json.dumps(errors))
for d in (W, pathlib.Path("work/mg_data"), pathlib.Path("work/mg13_data")):
    shutil.rmtree(d, ignore_errors=True)
open(OUT / "DONE", "w").write("ok")
