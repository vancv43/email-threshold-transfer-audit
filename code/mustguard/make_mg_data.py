"""Inputs of the MuST-Guard kernels (PROTOCOL.md section 3, A3, A4, A5, A8): one row per message with the four
branch inputs, the G1 paraphrases usable for training, and the two G2 sets with the same branches.

Branches
- body: subject + newline + body clipped to 5000 characters, the same text as every text baseline (A3).
- header: only from_name, from_domain and content_type (A4); the subject is already in the body text.
- url: the first 8 URLs joined by spaces (char-CNN input) and 11 lexical features.
- struct: n_forms, n_scripts, n_imgs, n_links, n_hidden, has_html.
A G1 or G2 rewrite replaces the body; subject, header, URLs and structure come from the original message
(the A3 filter keeps a rewrite only if every input URL is still in it).
The checks at the end stop the kernel before any training if a split, rewrite or header rule is broken.
Usage: python make_mg_data.py <messages.parquet> <splits.parquet> <g1_kept.parquet> <g2_kept.parquet> <out_dir>
"""
import ipaddress, json, pathlib, sys
from urllib.parse import urlsplit
import numpy as np
import pandas as pd

msg_p, splits_p, g1_p, g2_p, out = sys.argv[1:6]
out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
HEADER_FIELDS = ("from_name", "from_domain", "content_type")
STRUCT_FIELDS = ("n_forms", "n_scripts", "n_imgs", "n_links", "n_hidden", "has_html")
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
              "shorturl.at", "tiny.cc", "rb.gy", "lnkd.in", "s.id"}
N_URL, URL_CHARS = 8, 512

df = pd.read_parquet(msg_p)
df["year"] = df["year_month"].str[:4].astype(int)
hdr = df["header"].map(json.loads)
df["subject"] = hdr.map(lambda h: h.get("subject", "") or "")
df["text"] = df["subject"] + "\n" + df["body"].str[:5000]
fields = hdr.map(lambda h: [str(h.get(k, "") or "") for k in HEADER_FIELDS])
df["hdr_text"] = fields.map(lambda f: " | ".join(f"{k}:{v}" for k, v in zip(HEADER_FIELDS, f)))
df["has_hdr"] = fields.map(lambda f: int(any(v.strip() for v in f)))
urls = df["urls"].map(json.loads)
df["url_text"] = urls.map(lambda u: " ".join(u[:N_URL])[:URL_CHARS])
df["has_url"] = urls.map(lambda u: int(len(u) > 0))


def host_of(u):
    try:
        return (urlsplit(u if "://" in u else "http://" + u).hostname or "").lower()
    except ValueError:
        return ""


def is_ip(h):
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def url_lex(us):
    if not us:
        return [0.0] * 11
    hs = [host_of(u) for u in us]
    return [np.log1p(len(us)), np.log1p(np.mean([len(u) for u in us])), np.log1p(max(len(u) for u in us)),
            np.mean([is_ip(h) for h in hs]), np.mean([h in SHORTENERS for h in hs]),
            np.mean(["xn--" in h for h in hs]), np.log1p(len(set(hs))), np.mean([u.startswith("https") for u in us]),
            np.log1p(max(h.count(".") for h in hs)), np.mean(["@" in u for u in us]),
            np.mean([sum(c.isdigit() for c in h) / max(1, len(h)) for h in hs])]


df["url_lex"] = urls.map(lambda u: [float(x) for x in url_lex(u)])
df["struct_x"] = df["struct"].map(lambda s: [float(np.log1p(json.loads(s).get(k, 0))) for k in STRUCT_FIELDS])
COLS = ["msg_id", "label", "source", "year", "dup_cluster", "text", "hdr_text", "has_hdr", "url_text", "has_url",
        "url_lex", "struct_x"]
df[COLS].to_parquet(out / "mg.parquet", index=False)

sp = pd.read_parquet(splits_p)
sp.to_parquet(out / "splits.parquet", index=False)
base = df.set_index("msg_id")


def rewritten(rw):
    o = base.loc[rw["msg_id"]].reset_index()
    o["text"] = o["subject"].values + "\n" + rw["rewritten"].str[:5000].values
    o["label"] = 1
    return o[COLS]


g1 = pd.read_parquet(g1_p)
rewritten(g1).to_parquet(out / "g1_aug.parquet", index=False)  # filtered per seed to that seed's train split
s3_ids = set(sp.loc[(sp.seed == 0) & (sp.split == "s3_test"), "msg_id"])
s3 = df[df["msg_id"].isin(s3_ids)]
ham = s3[s3["label"] == 0][COLS]
g2 = pd.read_parquet(g2_p)
org = base.loc[g2["msg_id"]].reset_index()[COLS]
pd.concat([ham.assign(set="g2_rewrite"), rewritten(g2).assign(set="g2_rewrite"), ham.assign(set="g2_original"),
           org.assign(set="g2_original")]).to_parquet(out / "g2_sets.parquet", index=False)

# checks: stop here rather than train on a broken input
for seed, g in sp.groupby("seed"):
    assert not g["msg_id"].duplicated().any(), f"seed {seed}: a message is in two splits"
    tr = set(g.loc[g.split == "train", "msg_id"])
    held = set(g.loc[g.split != "train", "msg_id"])
    assert not tr & held
    assert set(g2["msg_id"]) <= set(g.loc[g.split == "s3_test", "msg_id"]), f"seed {seed}: G2 outside S3 test"
    assert not set(g1["msg_id"]) & set(g.loc[g.split.isin(["s2_test", "s3_test"]), "msg_id"]), "G1 in a test set"
assert not set(g1["msg_id"]) & set(g2["msg_id"])
assert set(g1["msg_id"]) <= set(base.index[base["label"] == 1]) and set(g2["msg_id"]) <= set(base.index[base["label"] == 1])
# A4: the header branch is built from these three fields only (values may contain any text, e.g. a domain "...spf...")
assert HEADER_FIELDS == ("from_name", "from_domain", "content_type"), "A4 header fields changed"
assert df["hdr_text"].str.startswith("from_name:").all() and df["hdr_text"].str.count(r"\| from_domain:").ge(1).all()
stats = {"rows": len(df), "g1_rows": len(g1), "g2_rows": len(g2), "s3_ham": len(ham),
         "has_hdr_by_source": df.groupby("source")["has_hdr"].mean().round(4).to_dict(),
         "has_url_by_source": df.groupby("source")["has_url"].mean().round(4).to_dict()}
(out / "mg_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats), flush=True)
