"""Make the locked splits of PROTOCOL.md v0.2 from data/processed/messages.parquet.

Training pool: untroubled spam, Nazario phishing and the list ham (Apache, GNU, Python, Fedora; A7) dated 2015-2020.
Per seed, near-duplicate clusters of that pool are assigned to train 70%, val 10%, cal 10%, S1 test 10%.
S2 test: SpamAssassin and Enron-Spam, never used for training.
S3 test: the same three modern sources dated 2021-2026; clusters seen in the training pool are removed.
Rewrite inputs: G1 paraphrases come from training-pool spam only, G2 rewrites from S3 test spam only.
"""
import hashlib, json, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"
SEEDS = range(5)
N_G1, N_G2 = 3000, 1500


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def main():
    df = pd.read_parquet(P / "messages.parquet")
    df["year"] = df["year_month"].str[:4].astype(int)
    modern = df[df["source"].isin(["untroubled", "nazario", "apache", "gnu", "python", "fedora"])]
    pool = modern[modern["year"].between(2015, 2020)]
    s3 = modern[modern["year"].between(2021, 2026)]
    s3 = s3[~s3["dup_cluster"].isin(set(pool["dup_cluster"]))]
    s2 = df[df["source"].isin(["spamassassin", "enron_spam"])]
    s2 = s2[~s2["dup_cluster"].isin(set(pool["dup_cluster"]))]

    rows = []
    for seed in SEEDS:
        r = pool["dup_cluster"].map(lambda c: h(f"{seed}:{c}") % 100)
        split = pd.cut(r, [-1, 69, 79, 89, 99], labels=["train", "val", "cal", "s1_test"]).astype(str)
        rows.append(pd.DataFrame({"msg_id": pool["msg_id"], "seed": seed, "split": split}))
        rows.append(pd.DataFrame({"msg_id": s2["msg_id"], "seed": seed, "split": "s2_test"}))
        rows.append(pd.DataFrame({"msg_id": s3["msg_id"], "seed": seed, "split": "s3_test"}))
    splits = pd.concat(rows, ignore_index=True)
    splits.to_parquet(P / "splits.parquet", index=False)

    spam_pool = pool[pool["label"] == 1]
    g1 = spam_pool.assign(k=spam_pool["msg_id"].map(lambda m: h("g1:" + m))).nsmallest(N_G1, "k")
    s3_spam = s3[s3["label"] == 1]
    g2 = s3_spam.assign(k=s3_spam["msg_id"].map(lambda m: h("g2:" + m))).nsmallest(N_G2, "k")
    g1[["msg_id", "source", "year", "body"]].to_parquet(P / "rewrite_g1_input.parquet", index=False)
    g2[["msg_id", "source", "year", "body"]].to_parquet(P / "rewrite_g2_input.parquet", index=False)

    def digest(ids):
        return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()

    man = {"protocol": "v0.2+A7", "counts": {}, "sha256": {}}
    for (seed, sp), g in splits.groupby(["seed", "split"]):
        lab = df.set_index("msg_id").loc[g["msg_id"], "label"]
        man["counts"][f"seed{seed}/{sp}"] = {"ham": int((lab == 0).sum()), "spam": int((lab == 1).sum())}
        man["sha256"][f"seed{seed}/{sp}"] = digest(g["msg_id"])
    man["rewrite_inputs"] = {"g1": {"n": len(g1), "sha256": digest(g1["msg_id"])},
                             "g2": {"n": len(g2), "sha256": digest(g2["msg_id"])}}
    man["s3_by_year"] = {str(y): {"ham": int(((s3.year == y) & (s3.label == 0)).sum()),
                                  "spam": int(((s3.year == y) & (s3.label == 1)).sum())} for y in sorted(s3.year.unique())}
    (P / "split_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man["counts"].items() if k.startswith("seed0")}, indent=1))
    print(json.dumps(man["s3_by_year"], indent=1))


if __name__ == "__main__":
    main()
