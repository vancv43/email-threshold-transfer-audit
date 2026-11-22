"""A13: target-site adaptation split. Per seed, the S2 ham is divided by corpus into 5,000 adaptation ham (moved to
train, label 0), 5,000 target-calibration ham (split tcal), and the rest, which stays in s2_test with all S2 spam.
No S2 spam leaves s2_test. Usage: python a13_split.py <messages.parquet or mg.parquet> <splits in> <splits out>
"""
import json, sys
import numpy as np
import pandas as pd

N_ADAPT, N_CAL = 5000, 5000  # A13, fixed in advance


def apply(sp, msg, n_adapt=N_ADAPT, n_cal=N_CAL):
    out = sp.copy()
    m = sp.merge(msg[["msg_id", "label", "source"]], on="msg_id", how="left")
    assert m["source"].notna().all(), "split rows without a message"
    for seed in sorted(sp["seed"].unique()):
        rng = np.random.default_rng(3000 + int(seed))
        ham = m[(m["seed"] == seed) & (m["split"] == "s2_test") & (m["label"] == 0)]
        share = ham["source"].value_counts(normalize=True)
        for src, frac in share.items():  # stratified by corpus
            idx = ham.index[ham["source"] == src].to_numpy(); rng.shuffle(idx)
            na, nc = int(round(n_adapt * frac)), int(round(n_cal * frac))
            out.loc[idx[:na], "split"] = "train"
            out.loc[idx[na:na + nc], "split"] = "tcal"
    chk = out.merge(msg[["msg_id", "label", "source"]], on="msg_id")
    s2src = {"spamassassin", "enron_spam"}
    assert not ((chk["split"] == "train") & (chk["label"] == 1) & chk["source"].isin(s2src)).any(), "S2 spam in train"
    assert (chk.loc[chk["split"] == "tcal", "label"] == 0).all()
    for k in ["s1_test", "s3_test", "val", "cal"]:
        assert (sp["split"] == k).sum() == (out["split"] == k).sum()
    stats = chk.groupby(["seed", "split", "label"]).size().rename("n").reset_index()
    return out, stats


if __name__ == "__main__":
    msg = pd.read_parquet(sys.argv[1], columns=["msg_id", "label", "source"])
    out, stats = apply(pd.read_parquet(sys.argv[2]), msg)
    out.to_parquet(sys.argv[3], index=False)
    print(json.dumps(stats[stats.seed == 0].to_dict("records")))
