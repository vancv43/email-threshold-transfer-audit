"""Message table of amendment A7: the frozen A1 table plus the messages of the three new ham hosts.

The A1 sources are not rebuilt from a new download, because the hosts changed some files since A1 (see
dl_check.json); the A1 rows are kept as they are. New rows that repeat an A1 body are dropped, and so are new rows
whose body appears with the other label (A1 rows are never removed). A1 rows keep their A1 near-duplicate
cluster, so the A1 splits and rewrite inputs do not move. A new row joins the cluster of its first near-duplicate
by row order, with the same MinHash settings as build_dataset.py, or starts its own cluster. Hits are sorted
because the LSH query returns them in hash-seed dependent order.
Usage: python merge_a7.py   (reads data/processed_A1/ and data/processed_A7_new/, writes data/processed/)
"""
import hashlib, json, pathlib
import pandas as pd
from datasketch import MinHash, MinHashLSH

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW_SOURCES = ["gnu", "python", "fedora"]
a1 = pd.read_parquet(ROOT / "data/processed_A1/messages.parquet")
new = pd.read_parquet(ROOT / "data/processed_A7_new/messages.parquet")
new = new[new["source"].isin(NEW_SOURCES)].drop(columns=["dup_cluster"])
n_new = len(new)
seen = a1.set_index("body_sha1")["label"]
same = new["body_sha1"].isin(seen.index)
conflict = same & (new["body_sha1"].map(seen).fillna(-1) != new["label"])
new = new[~same]
new = new[~new.duplicated(["body_sha1"])]  # within the new hosts all rows are ham
df = pd.concat([a1, new], ignore_index=True)
lsh, cluster = MinHashLSH(threshold=0.8, num_perm=128), {}
norm = df["body"].str.lower().str.replace(r"\s+", " ", regex=True).str[:5000]
for i, s in enumerate(norm):
    mh = MinHash(num_perm=128, seed=1)
    mh.update_batch([s[j:j + 5].encode() for j in range(max(1, len(s) - 4))])
    if i < len(a1):
        cluster[i] = int(df.at[i, "dup_cluster"])
    else:
        hits = sorted(int(k) for k in lsh.query(mh))
        cluster[i] = cluster[hits[0]] if hits else i
    lsh.insert(str(i), mh)
df["dup_cluster"] = [cluster[i] for i in range(len(df))]
out = ROOT / "data/processed"; out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "messages.parquet", index=False)
stats = {"a1_rows": int(len(a1)), "new_rows_parsed": int(n_new), "new_dropped_same_body_as_A1": int(same.sum()),
         "of_which_other_label": int(conflict.sum()), "rows": int(len(df)),
         "near_dup_clusters": int(df["dup_cluster"].nunique()),
         "by_source_label": {f"{s}|{l}": int(n) for (s, l), n in df.groupby(["source", "label"]).size().items()},
         "by_source_year": {f"{s}|{y}": int(n) for (s, y), n in
                            df.assign(y=df["year_month"].str[:4]).groupby(["source", "y"]).size().items()}}
(out / "build_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if k != "by_source_year"}, indent=1))
