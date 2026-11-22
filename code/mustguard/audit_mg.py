"""Code audit of MuST-Guard before any the GPU server run (A8). Not an experiment: a tiny test encoder
(hf-internal-testing/tiny-random-bert, random weights) is trained for one epoch on CPU on a few hundred real A1 messages, only to show that every
step runs and that the leakage and protocol checks hold.

Checks
1. make_mg_data.py on a synthetic split: inputs built, A4 header rule, rewrite and split checks pass; a broken
   split (G2 message moved to train) must make it fail.
2. obfuscate(): subject and URLs unchanged, deterministic for a fixed RNG, applied to the body only.
3. model: a missing branch gets gate weight 0; the body branch alone works.
4. train_mg.py tune + infer, pilot mode (no infer allowed), G1 rows limited to this seed's training spam.
5. score_mg.py: conformal coverage >= 1 - alpha on held-out data; threshold FPR on cal <= 0.1%.
Usage: python audit_mg.py <processed_A1/messages.parquet> <g1_kept.parquet> <g2_kept.parquet> <tmp_dir>
"""
import json, pathlib, random, shutil, subprocess, sys
import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from model import MuSTGuard, encode_bytes
from obfuscate import obfuscate

msg_p, g1_p, g2_p, tmp = sys.argv[1:5]
T = pathlib.Path(tmp); shutil.rmtree(T, ignore_errors=True); T.mkdir(parents=True)
PY, TINY = sys.executable, "hf-internal-testing/tiny-random-bert"
ok = {}

# 1. data builder on a small synthetic split of real messages
m = pd.read_parquet(msg_p)
g1, g2 = pd.read_parquet(g1_p).head(40), pd.read_parquet(g2_p).head(40)
ham = m[(m.label == 0) & (m.source == "apache")].sample(240, random_state=0)
spam = m[(m.label == 1) & (m.source == "untroubled")].sample(160, random_state=0)
enron = m[m.source == "enron_spam"].sample(60, random_state=0)
sub = pd.concat([ham, spam, enron, m[m.msg_id.isin(g1.msg_id) | m.msg_id.isin(g2.msg_id)]]).drop_duplicates("msg_id")
sub.to_parquet(T / "messages.parquet", index=False)
rng = np.random.RandomState(0)
rest = sub[~sub.msg_id.isin(g2.msg_id) & ~sub.msg_id.isin(g1.msg_id) & (sub.source != "enron_spam")].msg_id.values
rng.shuffle(rest)
n = len(rest)
spl = {**{x: "train" for x in rest[: n // 2]}, **{x: "val" for x in rest[n // 2: 5 * n // 8]},
       **{x: "cal" for x in rest[5 * n // 8: 6 * n // 8]}, **{x: "s1_test" for x in rest[6 * n // 8: 7 * n // 8]},
       **{x: "s3_test" for x in rest[7 * n // 8:]}, **{x: "train" for x in g1.msg_id},
       **{x: "s3_test" for x in g2.msg_id}, **{x: "s2_test" for x in enron.msg_id}}
sp = pd.DataFrame({"msg_id": list(spl), "seed": 0, "split": list(spl.values())})
sp.to_parquet(T / "splits.parquet", index=False)
g1.to_parquet(T / "g1.parquet", index=False); g2.to_parquet(T / "g2.parquet", index=False)
D = T / "data"
r = subprocess.run([PY, str(HERE / "make_mg_data.py"), str(T / "messages.parquet"), str(T / "splits.parquet"),
                    str(T / "g1.parquet"), str(T / "g2.parquet"), str(D)], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
mg = pd.read_parquet(D / "mg.parquet")
assert not mg.hdr_text.str.contains(r"(?:^| \| )(?:subject|spf|dkim|dmarc|reply_to_mismatch):").any()
assert mg.loc[mg.source == "enron_spam", "has_url"].sum() == 0
g2s = pd.read_parquet(D / "g2_sets.parquet")
assert set(g2s.loc[g2s.label == 1, "msg_id"]) == set(g2.msg_id)
rw = g2s[(g2s.set == "g2_rewrite") & (g2s.label == 1)].set_index("msg_id")
og = g2s[(g2s.set == "g2_original") & (g2s.label == 1)].set_index("msg_id").loc[rw.index]
assert (rw.hdr_text == og.hdr_text).all() and (rw.url_text == og.url_text).all() and (rw.text != og.text).all()
ok["1_data_builder"] = True
bad = sp.copy(); bad.loc[bad.msg_id == g2.msg_id.iloc[0], "split"] = "train"
bad.to_parquet(T / "bad_splits.parquet", index=False)
r = subprocess.run([PY, str(HERE / "make_mg_data.py"), str(T / "messages.parquet"), str(T / "bad_splits.parquet"),
                    str(T / "g1.parquet"), str(T / "g2.parquet"), str(T / "bad")], capture_output=True, text=True)
assert r.returncode != 0 and "G2 outside S3 test" in r.stderr
ok["1_broken_split_rejected"] = True

# 2. obfuscation
t = "Win a prize now\nDear customer, please verify your account at http://bad.example.com/login today"
o1, o2 = obfuscate(t, random.Random(1), rate=1.0), obfuscate(t, random.Random(1), rate=1.0)
assert o1 == o2 and o1 != t and o1.split("\n")[0] == "Win a prize now" and "http://bad.example.com/login" in o1
ok["2_obfuscation"] = True

# 3. gates of missing branches
mdl = MuSTGuard(TINY, dtype=torch.float32).eval()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TINY)
enc = dict(tok(["hello world", "buy now"], return_tensors="pt", padding=True))
args = (encode_bytes(["from_name:a", ""], 128), encode_bytes(["http://x.com", ""], 512), torch.zeros(2, 11),
        torch.zeros(2, 6))
with torch.no_grad():
    _, a = mdl(enc, *args, torch.tensor([[1., 1., 1., 1.], [1., 0., 0., 1.]]))
    _, a_body = MuSTGuard(TINY, use=("body",), dtype=torch.float32).eval()(enc, *args, torch.ones(2, 4))
assert a[1, 1] == 0 and a[1, 2] == 0 and torch.allclose(a.sum(1), torch.ones(2))
assert torch.allclose(a_body[:, 0], torch.ones(2))
ok["3_gates"] = True

# 4. tune + infer + pilot on CPU
base = [PY, str(HERE / "train_mg.py"), "--model", TINY, "--seed", "0", "--epochs", "1", "--batch", "16",
        "--max_len", "64", "--data", str(D), "--out", str(T / "out")]
r = subprocess.run(base + ["--tag", "t", "--ckpt", str(T / "t.pt")], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]
lg = json.loads((T / "out/log_t.json").read_text())
n_tr_spam_g1 = len(set(g1.msg_id) & set(sp.loc[sp.split == "train", "msg_id"]))
assert lg["n_para"] == n_tr_spam_g1, (lg["n_para"], n_tr_spam_g1)
r = subprocess.run(base + ["--tag", "t_best", "--ckpt", str(T / "t.pt"), "--infer"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]
sc = pd.read_parquet(T / "out/scores_t_best.parquet")
assert set(sc.set) == {"val", "cal", "s1_test", "s2_test", "s3_test", "g2_original", "g2_rewrite"}
assert np.allclose(sc[["gate_body", "gate_header", "gate_url", "gate_struct"]].sum(1), 1, atol=1e-3)
assert (sc.loc[sc.set == "s2_test", "gate_url"] == 0).all()  # Enron has no URLs
r = subprocess.run(base + ["--tag", "p", "--ckpt", str(T / "p.pt"), "--pilot_frac", "0.1", "--infer"],
                   capture_output=True, text=True)
assert r.returncode != 0 and "pilot runs never score" in r.stderr
r = subprocess.run(base + ["--tag", "p", "--ckpt", str(T / "p.pt"), "--pilot_frac", "0.5", "--para", "0"],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]
assert json.loads((T / "out/log_p.json").read_text())["n_train"] < lg["n_train"]
ok["4_train_infer_pilot"] = True

# 5. conformal and threshold on synthetic scores
from score_mg import conformal, conformal_logit, fit_temperature, sig
from common import threshold_at_fpr
rs = np.random.RandomState(0)
y = rs.randint(0, 2, 40000); s = rs.normal(2 * y - 1, 1.5)
for al in (0.01, 0.05):
    c = conformal(y[:20000], sig(s[:20000]), y[20000:], sig(s[20000:]), al)
    assert c["coverage"] >= 1 - al - 0.005, c
    c9 = conformal_logit(y[:20000], s[:20000], y[20000:], s[20000:], al)  # A9 = A8 without saturation, q differs in scale
    assert all(c9[k] == c[k] for k in ("coverage", "abstain", "accepted_tpr", "accepted_fpr"))
    z = 400 * rs.normal(2 * y - 1, 0.3)  # saturated sigmoid, as seed 4 of the confirmation (A9)
    assert conformal_logit(y[:20000], z[:20000], y[20000:], z[20000:], al)["coverage"] >= 1 - al - 0.005
thr = threshold_at_fpr(y[:20000], s[:20000], 0.001)
assert (s[:20000][y[:20000] == 0] > thr).mean() <= 0.001
assert 0.5 < fit_temperature(y, 3 * s) / 3 < 2
ok["5_conformal_threshold"] = True
print(json.dumps(ok))
