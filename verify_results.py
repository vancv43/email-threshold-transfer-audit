"""Check that every table of the paper follows from the result files in this repository.
Runs the analysis scripts on the stored score files, which rewrites results/tables/ and the result summaries, and
compares each regenerated file with the published copy byte for byte. Writes results/verification_report.md.
Usage: python verify_results.py   (about 10 minutes on a laptop; no training, no GPU)"""
import filecmp, pathlib, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent
STEPS = ["code/report/extra_analysis.py", "code/report/manuscript_tables.py", "code/analysis/jisa_tables.py",
         "code/analysis/jisa_revision.py", "code/analysis/a13_analysis.py", "code/analysis/a13_tables.py",
         "code/analysis/a14_analysis.py", "code/analysis/a14_tables.py"]
SKIP = {"tab_amend.tex", "tab_flow.tex"}
ref = pathlib.Path(tempfile.mkdtemp()) / "published"
shutil.copytree(ROOT / "results/tables", ref)
t0, log = time.time(), []
for s in STEPS:
    r = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT, capture_output=True, text=True)
    log.append(f"{s}: exit {r.returncode}")
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:]); sys.exit(f"{s} failed")
rows, bad = [], 0
for p in sorted(ref.glob("*.tex")):
    new = ROOT / "results/tables" / p.name
    if p.name in SKIP:
        rows.append(f"| {p.name} | written by hand or from data files not in this repository | not regenerated |"); continue
    same = new.exists() and filecmp.cmp(p, new, shallow=False)
    bad += not same
    rows.append(f"| {p.name} | regenerated from the score files | {'identical' if same else 'DIFFERENT'} |")
    if not same:
        shutil.copy(p, new)  # keep the published copy in place
rep = ["# Verification report", "", f"Analysis scripts run in {time.time() - t0:.0f} s: " + "; ".join(log), "",
       "| Table body | Source | Result |", "|---|---|---|"] + rows + ["",
       f"{len(rows) - bad - len(SKIP & {p.name for p in ref.glob('*.tex')})} regenerated tables identical, {bad} different."]
(ROOT / "results/verification_report.md").write_text("\n".join(rep) + "\n")
print("\n".join(rep[-1:])); sys.exit(1 if bad else 0)
