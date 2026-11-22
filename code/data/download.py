"""Download the public corpora listed in PROTOCOL.md (amendment A1). Idempotent: skips files already present."""
import pathlib, time, urllib.request, json, hashlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "academic-research-download (email-threat study)"}

def get(url, dest, pause=1.0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    time.sleep(pause)
    return "ok"

def jobs():
    for y in range(2015, 2026):
        yield f"http://untroubled.org/spam/{y}.7z", RAW / "untroubled" / f"{y}.7z"
    for m in range(1, 11):
        yield f"http://untroubled.org/spam/2026-{m:02d}.7z", RAW / "untroubled" / f"2026-{m:02d}.7z"
    for y in range(2015, 2026):
        yield f"https://monkey.org/~jose/phishing/phishing-{y}", RAW / "nazario" / f"phishing-{y}.mbox"
    yield "https://monkey.org/~jose/phishing/LICENSE.txt", RAW / "nazario" / "LICENSE.txt"
    lists = ["dev@httpd.apache.org", "user@spark.apache.org", "users@kafka.apache.org",
             "users@tomcat.apache.org", "dev@commons.apache.org"]
    for lst in lists:
        for y in range(2015, 2027):
            for m in (1, 4, 7, 10):
                if (y, m) > (2026, 7):
                    continue
                yield (f"https://lists.apache.org/api/mbox.lua?list={lst}&date={y}-{m:02d}",
                       RAW / "apache" / lst.replace("@", "_at_") / f"{y}-{m:02d}.mbox")
    # amendment A7: three more mailing-list hosts, same months as Apache
    months = [(y, m) for y in range(2015, 2027) for m in (1, 4, 7, 10) if (y, m) <= (2026, 7)]
    for lst in ["help-gnu-emacs", "bug-bash", "info-gnu"]:
        for y, m in months:
            yield f"https://lists.gnu.org/archive/mbox/{lst}/{y}-{m:02d}", RAW / "gnu" / lst / f"{y}-{m:02d}.mbox"
    for host, src, lists in [("mail.python.org", "python", ["python-list@python.org", "python-ideas@python.org",
                                                             "python-announce-list@python.org"]),
                             ("lists.fedoraproject.org", "fedora", ["users@lists.fedoraproject.org",
                                                                     "devel@lists.fedoraproject.org",
                                                                     "announce@lists.fedoraproject.org"])]:
        for lst in lists:
            for y, m in months:
                yield (f"https://{host}/archives/list/{lst}/export/x.mbox.gz?start={y}-{m:02d}-01"
                       f"&end={y + (m == 12)}-{m % 12 + 1:02d}-01",
                       RAW / src / lst.replace("@", "_at_") / f"{y}-{m:02d}.mbox.gz")
    base = "https://spamassassin.apache.org/old/publiccorpus/"
    for f in ["20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2", "20030228_hard_ham.tar.bz2",
              "20030228_spam.tar.bz2", "20050311_spam_2.tar.bz2"]:
        yield base + f, RAW / "spamassassin" / f
    for i in range(1, 7):
        yield (f"https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{i}.tar.gz",
               RAW / "enron_spam" / f"enron{i}.tar.gz")

if __name__ == "__main__":
    log = []
    for url, dest in jobs():
        for attempt in range(4):  # archives of the A7 hosts drop connections now and then
            try:
                st = get(url, dest)
                break
            except Exception as e:
                st = f"ERR {e}"
                time.sleep(20 * (attempt + 1))
        size = dest.stat().st_size if dest.exists() else 0
        print(f"{st}\t{size}\t{dest.relative_to(RAW)}", flush=True)
        log.append({"url": url, "file": str(dest.relative_to(RAW)), "status": st, "bytes": size})
    manifest = []
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and not p.name.endswith(".part"):
            manifest.append({"file": str(p.relative_to(RAW)), "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (RAW / "download_manifest.json").write_text(json.dumps({"downloads": log, "files": manifest}, indent=1))
    print("DONE", len(manifest), "files")
