"""Fine-tune one transformer text baseline of PROTOCOL.md (A3, A6) on one seed.

Tune mode trains one configuration, keeps the epoch with the best validation PR-AUC, saves that checkpoint and
writes a log. Infer mode loads a saved checkpoint and scores val, cal, S1, S2, S3 and the G2 stress sets.
Input text is subject + newline + body.
A11: --select ood picks the epoch by TPR at FPR 0.1% on val spam plus the held-out ham source (split ood_ham),
ties broken by in-domain val PR-AUC.
"""
import argparse, json, pathlib, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--infer", action="store_true")
ap.add_argument("--select", default="pr", choices=["pr", "ood"])  # A11 T1, same rule as MuST-Guard v2
a = ap.parse_args()
DATA, OUT = pathlib.Path(a.data), pathlib.Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for the A11 smoke test
amp = dev == "cuda"

msg = pd.read_parquet(DATA / "text.parquet")
sp = pd.read_parquet(DATA / "splits.parquet")
sp = sp[sp["seed"] == a.seed].merge(msg, on="msg_id")
part = {k: g.reset_index(drop=True) for k, g in sp.groupby("split")}
tok = AutoTokenizer.from_pretrained(a.model)
# fp32 master weights: the DeBERTa-v3 checkpoint loads as fp16 by default, which GradScaler cannot unscale
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, dtype=torch.float32).to(dev)


def batches(texts, bs):
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        yield {k: v.to(dev) for k, v in enc.items()}


@torch.no_grad()
def predict(texts):
    model.eval()
    order = np.argsort([len(t) for t in texts])  # length-sorted batches run faster
    out = []
    for enc in batches([texts[i] for i in order], 128):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg = model(**enc).logits.float()
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    s = np.empty(len(texts)); s[order] = np.concatenate(out)
    return s


if a.infer:
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    evals = {k: part[k] for k in ["val", "cal", "s1_test", "s2_test", "s3_test", "ood_ham", "tcal"] if k in part}  # A13: target-calibration ham
    for name, g in pd.read_parquet(DATA / "g2_sets.parquet").groupby("set"):
        evals[name] = g.reset_index(drop=True)
    t0, rows = time.time(), []
    for name, d in evals.items():
        rows.append(pd.DataFrame({"set": name, "msg_id": d["msg_id"], "label": d["label"],
                                  "score": predict(list(d["text"]))}))
    pd.concat(rows).to_parquet(OUT / f"scores_{a.tag}.parquet", index=False)
    print(json.dumps({"tag": a.tag, "infer_s": round(time.time() - t0, 1)}), flush=True)
    raise SystemExit

tr, va = part["train"], part["val"]
if a.select == "ood":
    held = part["ood_ham"]["source"].unique()
    assert len(held) == 1 and not ((tr["source"] == held[0]) & (tr["label"] == 0)).any(), "held-out ham in train"
    va_ood = pd.concat([va[va["label"] == 1], part["ood_ham"]], ignore_index=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler(enabled=amp)
log = {**vars(a), "n_train": int(len(tr)), "epoch_log": []}
best, t0 = (-1.0,), time.time()
for ep in range(a.epochs):
    model.train()
    perm = np.random.permutation(len(tr))
    texts, labels = tr["text"].values[perm], tr["label"].values[perm]
    for i, enc in enumerate(batches(texts, a.batch)):
        y = torch.tensor(labels[i * a.batch:(i + 1) * a.batch], device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
    v = float(average_precision_score(va["label"], predict(list(va["text"]))))
    e = {"epoch": ep + 1, "val_pr_auc": v, "elapsed_s": round(time.time() - t0, 1)}
    key = (v,)  # v1: the best validation PR-AUC
    if a.select == "ood":
        fo, to, _ = roc_curve(va_ood["label"], predict(list(va_ood["text"])))
        e["ood_tpr_fpr_0p1"] = float(to[fo <= 0.001].max())
        key = (e["ood_tpr_fpr_0p1"], v)
    log["epoch_log"].append(e)
    print(e, flush=True)
    if key > best:
        best, best_e = key, e
        torch.save(model.state_dict(), a.ckpt)
log.update(val_pr_auc=best_e["val_pr_auc"], sel_key=list(best), best_epoch=best_e["epoch"], train_s=round(time.time() - t0, 1))
(OUT / f"log_{a.tag}.json").write_text(json.dumps(log, indent=1))
print(json.dumps({"tag": a.tag, "sel_key": list(best), "train_s": log["train_s"]}), flush=True)
