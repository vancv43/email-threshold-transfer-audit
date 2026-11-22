"""Inputs of the transformer kernels (PROTOCOL.md A3, A5, A6): one text table, the splits and the two G2 sets.
Usage: python make_gpu_data.py <out_dir>"""
import pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import P, ROOT, load_messages

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
df, _ = load_messages()
sp = pd.read_parquet(P / "splits.parquet")
df[["msg_id", "label", "source", "year", "text"]].to_parquet(out / "text.parquet", index=False)
sp.to_parquet(out / "splits.parquet", index=False)
s3 = df[df["msg_id"].isin(sp.loc[(sp.seed == 0) & (sp.split == "s3_test"), "msg_id"])]  # S3 is the same for all seeds
ham = s3[s3["label"] == 0][["msg_id", "label", "text"]]
g2 = pd.read_parquet(ROOT / "data/rewrites/g2/rewrite_g2_kept.parquet")
orig = s3.set_index("msg_id").loc[g2["msg_id"]]
rew = pd.DataFrame({"msg_id": g2["msg_id"].values, "label": 1,
                    "text": orig["subject"].values + "\n" + g2["rewritten"].str[:5000].values})
org = pd.DataFrame({"msg_id": g2["msg_id"].values, "label": 1, "text": orig["text"].values})
pd.concat([ham.assign(set="g2_rewrite"), rew.assign(set="g2_rewrite"), ham.assign(set="g2_original"),
           org.assign(set="g2_original")])[["set", "msg_id", "label", "text"]].to_parquet(out / "g2_sets.parquet", index=False)
print(len(df), len(ham), len(g2))
