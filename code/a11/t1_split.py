"""A11 T1: hold one ham source out of train, val and cal for every seed (split name ood_ham).

The held-out ham is used only to choose the epoch and the configuration, together with the val spam; test splits
are unchanged. Usage: python t1_split.py <messages.parquet or mg.parquet> <splits.parquet in> <splits.parquet out> [source]
"""
import json, sys
import pandas as pd

HOLDOUT = "fedora"  # A11 default


def apply(sp, msg, holdout=HOLDOUT):
    m = sp.merge(msg[["msg_id", "label", "source"]], on="msg_id", how="left")
    assert m["source"].notna().all(), "split rows without a message"
    move = (m["source"] == holdout) & (m["label"] == 0) & m["split"].isin(["train", "val", "cal"])
    out = sp.copy()
    out.loc[move.values, "split"] = "ood_ham"
    chk = out.merge(msg[["msg_id", "label", "source"]], on="msg_id")
    pool = chk[chk["split"].isin(["train", "val", "cal"])]
    assert not ((pool["source"] == holdout) & (pool["label"] == 0)).any(), "held-out ham left in train, val or cal"
    assert (chk.loc[chk["split"] == "ood_ham", "label"] == 0).all()
    for k in ["s1_test", "s2_test", "s3_test"]:  # tests untouched
        assert (sp["split"] == k).sum() == (out["split"] == k).sum()
    stats = chk.groupby(["seed", "split", "label"]).size().rename("n").reset_index()
    return out, stats


if __name__ == "__main__":
    msg = pd.read_parquet(sys.argv[1], columns=["msg_id", "label", "source"])
    sp = pd.read_parquet(sys.argv[2])
    out, stats = apply(sp, msg, sys.argv[4] if len(sys.argv) > 4 else HOLDOUT)
    out.to_parquet(sys.argv[3], index=False)
    print(json.dumps(stats[stats.seed == 0].to_dict("records")))
