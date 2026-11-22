"""Parse the raw corpora into one table of messages with four views, then remove exact and near duplicates.

Views follow PROTOCOL.md amendment A1: source-specific headers are dropped, the header view keeps only
sender-side fields, and mailing-list footers are cut from the body.
Output: data/processed/messages.parquet and data/processed/build_stats.json
"""
import email, email.policy, email.utils, gzip, hashlib, html, json, mailbox, pathlib, re, shutil, subprocess, sys, tarfile
from collections import Counter, defaultdict
from datetime import timezone

import pandas as pd
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW, OUT, TMP = ROOT / "data/raw", ROOT / "data/processed", ROOT / "data/tmp"
N_MONTH = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # cap per source, year and month
SEED = "jnsm-email-2026"
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
FOOTER_RE = re.compile(r"\n-{5,}\s*\nTo unsubscribe.*", re.S | re.I)
# Mailman footers of the A7 hosts: "____\n<list> mailing list ..." and the HyperKitty "-- \nhttps://.../mailman3/..."
MAILMAN_RE = re.compile(r"\n_{10,}\s*\n[^\n]*mailing list.*|\n-- ?\n\s*https?://\S*mailman3?/.*", re.S | re.I)
LIST_SOURCES = ("apache", "gnu", "python", "fedora")


def rank(key):
    return hashlib.sha256((SEED + key).encode()).hexdigest()


def auth(msg):
    a = " ".join(str(v) for v in (msg.get_all("Authentication-Results") or [])).lower()
    out = {}
    for k in ("spf", "dkim", "dmarc"):
        m = re.search(rf"\b{k}=(\w+)", a)
        out[k] = m.group(1) if m else "absent"
    return out


def parts(msg):
    plain, htmls = [], []
    for p in msg.walk():
        if p.is_multipart() or p.get_content_disposition() == "attachment":
            continue
        ct = p.get_content_type()
        try:
            txt = p.get_content()
        except Exception:
            try:
                txt = p.get_payload(decode=True).decode("latin-1", "replace")
            except Exception:
                continue
        if not isinstance(txt, str):
            continue
        if ct == "text/plain":
            plain.append(txt)
        elif ct == "text/html":
            htmls.append(txt)
    return "\n".join(plain), "\n".join(htmls)


def views(raw, source):
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    plain, htm = parts(msg)
    struct = {"n_forms": 0, "n_scripts": 0, "n_imgs": 0, "n_links": 0, "n_hidden": 0, "has_html": int(bool(htm))}
    hrefs = []
    if htm:
        soup = BeautifulSoup(htm, "html.parser")
        struct.update(n_forms=len(soup.find_all("form")), n_scripts=len(soup.find_all("script")),
                      n_imgs=len(soup.find_all("img")), n_links=len(soup.find_all("a")),
                      n_hidden=len(soup.select('[style*="display:none"],[style*="display: none"],[hidden]')))
        hrefs = [a.get("href", "") for a in soup.find_all("a") if a.get("href", "").lower().startswith("http")]
        if not plain.strip():
            plain = soup.get_text("\n")
    body = html.unescape(plain)
    if source in LIST_SOURCES:
        body = MAILMAN_RE.sub("", FOOTER_RE.sub("", body))
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    urls = sorted(set(URL_RE.findall(body)) | set(hrefs))
    frm, rep = msg.get("From", ""), msg.get("Reply-To", "")
    fdom = (email.utils.parseaddr(frm)[1].rsplit("@", 1)[-1]).lower()
    rdom = (email.utils.parseaddr(rep)[1].rsplit("@", 1)[-1]).lower() if rep else ""
    header = {"subject": str(msg.get("Subject", ""))[:300], "from_name": email.utils.parseaddr(frm)[0][:120],
              "from_domain": fdom, "reply_to_mismatch": int(bool(rdom) and rdom != fdom),
              "content_type": msg.get_content_type(), **auth(msg)}
    date = None
    try:
        d = email.utils.parsedate_to_datetime(msg.get("Date", ""))
        date = (d.astimezone(timezone.utc) if d.tzinfo else d).strftime("%Y-%m-%d")
    except Exception:
        pass
    return body, urls, header, struct, date


def iter_untroubled(cap):
    """Same sample as a full extraction (top `cap` names by hash per year and month), but only the sampled
    files are written to disk."""
    for arc in sorted((RAW / "untroubled").glob("*.7z")):
        work = TMP / arc.stem
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        if shutil.which("7z"):
            lst = subprocess.run(["7z", "l", "-slt", str(arc)], check=True, capture_output=True, text=True).stdout
            names = [l[7:] for l in lst.splitlines() if l.startswith("Path = ")][1:]
        else:  # hosts without p7zip
            import py7zr
            with py7zr.SevenZipFile(arc) as z:
                names = z.getnames()
        groups = defaultdict(list)
        for n in names:
            q = pathlib.PurePosixPath(n.replace("\\", "/"))
            if q.suffix == ".txt" and len(q.parts) >= 3:
                groups[(q.parts[-3], q.parts[-2])].append(n)
        keep = {(y, m): sorted(v, key=lambda n: rank(pathlib.PurePosixPath(n).name))[:cap] for (y, m), v in groups.items()}
        targets = [n for v in keep.values() for n in v]
        done = False
        if shutil.which("7z"):  # old p7zip refuses absolute paths: run inside the work folder
            (work / "list.txt").write_text("\n".join(targets))
            done = subprocess.run(["7z", "x", "-y", str(arc.resolve()), "@list.txt"], cwd=work,
                                  stdout=subprocess.DEVNULL).returncode == 0
        if not done:
            import py7zr
            with py7zr.SevenZipFile(arc) as z:
                z.extract(path=work, targets=targets)
        for (y, m), v in sorted(keep.items()):
            for n in v:
                yield "untroubled", 1, f"{y}-{m}", f"untroubled/{y}/{m}/{pathlib.PurePosixPath(n).name}", (work / n).read_bytes()
        shutil.rmtree(work)


def iter_mbox(folder, source, label, ym_from_name, cap):
    for f in sorted([*folder.rglob("*.mbox"), *folder.rglob("*.mbox.gz")]):
        if f.suffix == ".gz":  # HyperKitty exports, A7
            TMP.mkdir(parents=True, exist_ok=True)
            plain = TMP / "cur.mbox"
            plain.write_bytes(gzip.decompress(f.read_bytes()) if f.stat().st_size else b"")
        else:
            plain = f
        by_month = defaultdict(list)
        for i, m in enumerate(mailbox.mbox(str(plain), create=False)):
            raw = m.as_bytes()
            if b"FOLDER INTERNAL DATA" in raw[:2000]:
                continue
            ym = ym_from_name(f, m)
            by_month[ym].append((f"{f.relative_to(RAW)}#{i}", raw))
        for ym, items in sorted(by_month.items()):
            for mid, raw in sorted(items, key=lambda t: rank(t[0]))[:cap]:
                yield source, label, ym, mid, raw


def ym_nazario(f, m):
    try:
        return email.utils.parsedate_to_datetime(m.get("Date", "")).strftime("%Y-%m")
    except Exception:
        return f.stem.split("-")[1] + "-00"


def iter_spamassassin():
    for f in sorted((RAW / "spamassassin").glob("*.tar.bz2")):
        label = 1 if "spam" in f.name else 0
        with tarfile.open(f) as t:
            for mem in t.getmembers():
                if mem.isfile() and not mem.name.endswith("cmds"):
                    yield "spamassassin", label, "2002-00", f"spamassassin/{mem.name}", t.extractfile(mem).read()


def iter_enron():
    for f in sorted((RAW / "enron_spam").glob("*.tar.gz")):
        with tarfile.open(f) as t:
            for mem in t.getmembers():
                if mem.isfile() and mem.name.endswith(".txt"):
                    label = 1 if "/spam/" in mem.name else 0
                    yield "enron_spam", label, "2006-00", f"enron_spam/{mem.name}", t.extractfile(mem).read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    sources = [
        iter_untroubled(N_MONTH),
        iter_mbox(RAW / "nazario", "nazario", 1, ym_nazario, N_MONTH),
        iter_mbox(RAW / "apache", "apache", 0, lambda f, m: f.stem, N_MONTH),
        *[iter_mbox(RAW / s, s, 0, lambda f, m: f.name.split(".")[0], N_MONTH) for s in ("gnu", "python", "fedora")],
        iter_spamassassin(),
        iter_enron(),
    ]
    for it in sources:
        for source, label, ym, mid, raw in it:
            try:
                body, urls, header, struct, date = views(raw, source)
            except Exception:
                skipped[f"{source}:parse"] += 1
                continue
            if len(body) < 20:
                skipped[f"{source}:empty"] += 1
                continue
            # lone surrogates from badly encoded messages (seen in the A7 hosts) become "?"; A1 rows have none
            utf = lambda t: t.encode("utf-8", "replace").decode("utf-8")
            body, header = utf(body), {k: utf(v) if isinstance(v, str) else v for k, v in header.items()}
            urls = [utf(u) for u in urls]
            rows.append({"msg_id": mid, "source": source, "label": label, "year_month": ym, "date": date,
                         "body": body[:20000], "urls": json.dumps(urls[:50]), "header": json.dumps(header),
                         "struct": json.dumps(struct)})
    df = pd.DataFrame(rows)
    n0 = len(df)
    norm = df["body"].str.lower().str.replace(r"\s+", " ", regex=True)
    df["body_sha1"] = norm.map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    df = df[~df.duplicated(["body_sha1", "label"])].reset_index(drop=True)
    conflict = df[df.duplicated("body_sha1", keep=False)]  # same body with both labels
    df = df[~df["body_sha1"].isin(conflict["body_sha1"])].reset_index(drop=True)
    n1 = len(df)
    # near-duplicate clusters: MinHash on character 5-grams, Jaccard threshold 0.8
    lsh, cluster = MinHashLSH(threshold=0.8, num_perm=128), {}
    norm = df["body"].str.lower().str.replace(r"\s+", " ", regex=True).str[:5000]
    for i, s in enumerate(norm):
        mh = MinHash(num_perm=128, seed=1)
        mh.update_batch([s[j:j + 5].encode() for j in range(max(1, len(s) - 4))])
        hits = lsh.query(mh)
        cluster[i] = cluster[int(hits[0])] if hits else i
        lsh.insert(str(i), mh)
    df["dup_cluster"] = [cluster[i] for i in range(len(df))]
    df.to_parquet(OUT / "messages.parquet", index=False)
    stats = {"n_month_cap": N_MONTH, "parsed": n0, "after_exact_dedup_and_label_conflicts": n1,
             "label_conflicts_removed": int(conflict["body_sha1"].nunique()),
             "near_dup_clusters": int(df["dup_cluster"].nunique()), "skipped": dict(skipped),
             "by_source_label": {f"{s}|{l}": int(n) for (s, l), n in df.groupby(["source", "label"]).size().items()},
             "by_source_year": {f"{s}|{y}": int(n) for (s, y), n in
                                df.assign(y=df["year_month"].str[:4]).groupby(["source", "y"]).size().items()}}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
