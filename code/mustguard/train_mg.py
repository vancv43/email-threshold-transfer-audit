"""Train or score one MuST-Guard configuration on one seed (PROTOCOL.md section 3, A6 grid, A8).

Tune mode trains one configuration with the same schedule as the transformer baselines (AdamW, linear schedule,
6% warm-up, 3 epochs, fp16 autocast with fp32 master weights), keeps the epoch with the best validation PR-AUC,
saves that checkpoint and writes a log. Infer mode scores val, cal, S1, S2, S3 and the G2 sets once, and also
saves the fusion gate weights. Pilot mode trains on a fixed 10% of the training split and never scores test sets.
Robust training (curriculum C by default): G1 paraphrases of this seed's training spam are added as extra spam
rows, and each training row is obfuscated with probability 0.25 per epoch, ham and spam alike.
A11 (v2): --select ood picks the epoch by TPR at FPR 0.1% on val spam plus the held-out ham source (split ood_ham,
made by a11/t1_split.py), ties broken by in-domain val PR-AUC; --aux adds a body-only loss; --ls smooths labels.
"""
import argparse, hashlib, json, pathlib, random, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import MuSTGuard, encode_bytes
from obfuscate import obfuscate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--head_lr", type=float, default=1e-3)  # fixed for the new branches, not tuned
ap.add_argument("--para", type=int, default=1)  # G1 paraphrases (curricula B, C)
ap.add_argument("--obf", type=int, default=1)  # character obfuscation (curricula A, C)
ap.add_argument("--p_obf", type=float, default=0.25)
ap.add_argument("--mdrop", type=float, default=0.1)  # modality dropout of the non-body branches
ap.add_argument("--use", default="body,header,url,struct")
ap.add_argument("--aux", type=float, default=0.0)  # A11 T3: weight of the body-only loss
ap.add_argument("--ls", type=float, default=0.0)  # A11 T4: label smoothing
ap.add_argument("--select", default="pr", choices=["pr", "ood"])  # A11 T1: ood = held-out ham source
ap.add_argument("--pilot_frac", type=float, default=0.0)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--infer", action="store_true")
ap.add_argument("--init", default="")  # A14b: start from these weights
ap.add_argument("--freeze_gate", type=int, default=0)  # A14b: keep the gate network fixed
ap.add_argument("--drop_target_ham", type=int, default=0)  # A14b stage 1: remove the A13 adaptation ham from train
a = ap.parse_args()
assert "body" in a.use.split(","), "the body branch is always used"
assert not (a.infer and a.pilot_frac), "pilot runs never score the test sets"
DATA, OUT = pathlib.Path(a.data), pathlib.Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dev == "cuda"
HDR_CHARS, URL_CHARS = 128, 512

msg = pd.read_parquet(DATA / "mg.parquet")
sp = pd.read_parquet(DATA / "splits.parquet")
sp = sp[sp["seed"] == a.seed].merge(msg, on="msg_id")
part = {k: g.reset_index(drop=True) for k, g in sp.groupby("split")}
tok = AutoTokenizer.from_pretrained(a.model)
model = MuSTGuard(a.model, use=a.use.split(","), dtype=torch.float32).to(dev)


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def tensors(d, texts=None):
    texts = list(d["text"]) if texts is None else texts
    enc = tok(texts, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
    mask = np.stack([np.ones(len(d)), d["has_hdr"].values, d["has_url"].values, np.ones(len(d))], 1)
    return ({k: v.to(dev) for k, v in enc.items()}, encode_bytes(list(d["hdr_text"]), HDR_CHARS).to(dev),
            encode_bytes(list(d["url_text"]), URL_CHARS).to(dev),
            torch.tensor(np.stack(d["url_lex"].values), dtype=torch.float32, device=dev),
            torch.tensor(np.stack(d["struct_x"].values), dtype=torch.float32, device=dev),
            torch.tensor(mask, dtype=torch.float32, device=dev))


@torch.no_grad()
def predict(d):
    model.eval()
    order = np.argsort(d["text"].str.len().values)  # length-sorted batches run faster
    s, al = np.empty(len(d)), np.empty((len(d), 4))
    for i in range(0, len(d), 128):
        idx = order[i:i + 128]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg, w = model(*tensors(d.iloc[idx]))
        lg = lg.float()
        s[idx] = (lg[:, 1] - lg[:, 0]).cpu().numpy(); al[idx] = w.float().cpu().numpy()
    return s, al


if a.infer:
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    evals = {k: part[k] for k in ["val", "cal", "s1_test", "s2_test", "s3_test", "ood_ham", "tcal"] if k in part}  # A13: target-calibration ham
    for name, g in pd.read_parquet(DATA / "g2_sets.parquet").groupby("set"):
        evals[name] = g.reset_index(drop=True)
    t0, rows = time.time(), []
    for name, d in evals.items():
        s, al = predict(d)
        rows.append(pd.DataFrame({"set": name, "msg_id": d["msg_id"], "label": d["label"], "score": s,
                                  **{f"gate_{b}": al[:, j] for j, b in enumerate(MuSTGuard.BRANCHES)}}))
    pd.concat(rows).to_parquet(OUT / f"scores_{a.tag}.parquet", index=False)
    print(json.dumps({"tag": a.tag, "infer_s": round(time.time() - t0, 1)}), flush=True)
    raise SystemExit

tr, va = part["train"], part["val"]
if a.drop_target_ham:
    tr = tr[~tr["source"].isin({"spamassassin", "enron_spam"})].reset_index(drop=True)
n_target_ham = int(tr["source"].isin({"spamassassin", "enron_spam"}).sum())
assert not (tr["source"].isin({"spamassassin", "enron_spam"}) & (tr["label"] == 1)).any(), "target spam in train"
if a.init:
    model.load_state_dict(torch.load(a.init, map_location=dev))
if a.freeze_gate:
    for p_ in model.gate.parameters():
        p_.requires_grad_(False)
if a.select == "ood":
    held = part["ood_ham"]["source"].unique()
    assert len(held) == 1 and not ((tr["source"] == held[0]) & (tr["label"] == 0)).any(), "held-out ham in train"
    va_ood = pd.concat([va[va["label"] == 1], part["ood_ham"]], ignore_index=True)
if a.pilot_frac:
    tr = tr[tr["msg_id"].map(lambda m: h("pilot:" + m) % 1000 < a.pilot_frac * 1000)].reset_index(drop=True)
n_orig = len(tr)
if a.para:
    g1 = pd.read_parquet(DATA / "g1_aug.parquet")
    g1 = g1[g1["msg_id"].isin(set(tr["msg_id"]))]  # only paraphrases of this seed's training spam
    assert g1["label"].eq(1).all() and set(g1["msg_id"]) <= set(tr["msg_id"])
    tr = pd.concat([tr, g1[tr.columns.intersection(g1.columns)]], ignore_index=True)
log = {**vars(a), "n_train": n_orig, "n_para": len(tr) - n_orig, "n_target_ham": n_target_ham, "epoch_log": []}

enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
new_p = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": new_p, "lr": a.head_lr}], weight_decay=a.wd)
steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler(enabled=amp)
best, t0 = (-1.0,), time.time()
for ep in range(a.epochs):
    model.train()
    r = random.Random(f"{a.seed}:{ep}")
    perm = np.random.permutation(len(tr))
    for i in range(0, len(tr), a.batch):
        d = tr.iloc[perm[i:i + a.batch]]
        texts = [obfuscate(t, r) if a.obf and r.random() < a.p_obf else t for t in d["text"]]
        enc, hb, ub, lex, st, mask = tensors(d, texts)
        if a.mdrop:
            keep = (torch.rand(len(d), 3, device=dev) >= a.mdrop).float()
            mask = mask * torch.cat([torch.ones(len(d), 1, device=dev), keep], 1)
        y = torch.tensor(d["label"].values, device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg, _ = model(enc, hb, ub, lex, st, mask)
            loss = torch.nn.functional.cross_entropy(lg.float(), y, label_smoothing=a.ls)
            if a.aux:
                loss = loss + a.aux * torch.nn.functional.cross_entropy(model.aux_logits.float(), y, label_smoothing=a.ls)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
    s, _ = predict(va)
    v = float(average_precision_score(va["label"], s))
    fpr, tpr, _ = roc_curve(va["label"], s)
    v_tpr = float(tpr[fpr <= 0.001].max())
    e = {"epoch": ep + 1, "val_pr_auc": v, "val_tpr_fpr_0p1": v_tpr, "elapsed_s": round(time.time() - t0, 1)}
    key = (v,)  # v1: the best validation PR-AUC, as for the baselines
    if a.select == "ood":
        so, _ = predict(va_ood)
        fo, to, _ = roc_curve(va_ood["label"], so)
        e["ood_tpr_fpr_0p1"] = float(to[fo <= 0.001].max())
        key = (e["ood_tpr_fpr_0p1"], v)
    log["epoch_log"].append(e)
    print(e, flush=True)
    if key > best:
        best, best_e = key, e
        torch.save(model.state_dict(), a.ckpt)
log.update(val_pr_auc=best_e["val_pr_auc"], val_tpr_fpr_0p1=best_e["val_tpr_fpr_0p1"], sel_key=list(best),
           best_epoch=best_e["epoch"], train_s=round(time.time() - t0, 1))
(OUT / f"log_{a.tag}.json").write_text(json.dumps(log, indent=1))
print(json.dumps({"tag": a.tag, "sel_key": list(best), "train_s": log["train_s"]}), flush=True)
