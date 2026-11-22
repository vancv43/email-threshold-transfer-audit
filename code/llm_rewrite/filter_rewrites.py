"""Apply the filter of PROTOCOL.md A3 to the G1 and G2 rewrites.

Kept: no regex refusal, every input URL present verbatim in the output, and at least 90% of the input letters are
ASCII letters. Output: data/rewrites/<g>/rewrite_<g>_kept.parquet and results/rewrite_filter.json
"""
import json, pathlib, re
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)


def ascii_share(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


out = {}
for g in ["g1", "g2"]:
    r = pd.read_parquet(ROOT / f"data/rewrites/{g}/out/rewrite_{g}.parquet")
    i = pd.read_parquet(ROOT / f"data/processed/rewrite_{g}_input.parquet")
    m = i.merge(r, on="msg_id")
    links = m.apply(lambda x: set(URL_RE.findall(x.body)) <= set(URL_RE.findall(x.rewritten)), axis=1)
    english = m["body"].map(ascii_share) >= 0.9
    keep = ~m["refusal"] & links & english
    m[keep][["msg_id", "source", "year", "generator", "model", "rewritten"]].to_parquet(
        ROOT / f"data/rewrites/{g}/rewrite_{g}_kept.parquet", index=False)
    out[g] = {"n": int(len(m)), "refusal": int(m["refusal"].sum()), "links_lost": int((~links).sum()),
              "non_english": int((~english).sum()), "kept": int(keep.sum()),
              "kept_by_source": m[keep]["source"].value_counts().to_dict(),
              "kept_by_year": {str(k): int(v) for k, v in m[keep]["year"].value_counts().sort_index().items()}}
(ROOT / "results/rewrite_filter.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
