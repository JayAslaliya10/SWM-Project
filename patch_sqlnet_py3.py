from pathlib import Path
import re

root = Path("sqlnet")

def patch_file(p: Path, subs):
    s0 = p.read_text(encoding="utf-8")
    s = s0
    for pat, rep in subs:
        s = re.sub(pat, rep, s, flags=re.M)
    if s != s0:
        p.write_text(s, encoding="utf-8")
        print(f"patched: {p}")

# 1) Replace legacy "from modules." imports with "from sqlnet."
for f in root.rglob("*.py"):
    if f.name.endswith(".py.bak"):
        continue
    patch_file(f, [
        (r"\bfrom\s+modules\.", "from sqlnet."),
        (r"\bimport\s+modules\.", "import sqlnet."),
    ])

# 2) Fix stubborn Python-2 style prints in model/sqlnet.py (the usual offenders)
msql = root / "model" / "sqlnet.py"
if msql.exists():
    patch_file(msql, [
        (r"""print\s+'question:',\s*vis_data\[0\]""", r"""print('question:', vis_data[0])"""),
        (r"""print\s+'headers:\s*\(%s\)'\s*%\s*\(' \| '\.join\(vis_data\[1\]\)\)""",
         r"""print('headers: (%s)' % (' | '.join(vis_data[1])))"""),
        (r"""print\s+'query:',\s*vis_data\[2\]""", r"""print('query:', vis_data[2])"""),
        (r"""^\s*print\s+'([^']*)'\s*$""", r"""print('\1')"""),
    ])

print("✅ patch complete")