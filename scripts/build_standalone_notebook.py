"""Baut ein eigenständiges Notebook mit eingebettetem p229-Code.

Aufruf (Repo-Ordner): python scripts/build_standalone_notebook.py [quelle] [ziel]
Beispiel: python scripts/build_standalone_notebook.py db_10_modellvergleich db_10_modellvergleich_komplett
Für Umgebungen ohne Git und ohne Datei-Upload: nur dieses eine Notebook importieren.
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
import sys
FILES = ["__init__", "config", "stats", "features", "leakage", "walk_forward", "evaluate", "data", "models"]
QUELLE = sys.argv[1] if len(sys.argv) > 1 else "db_09_neuaufbau"
ZIEL = sys.argv[2] if len(sys.argv) > 2 else "db_09_komplett"

src = [l for l in (root / f"notebooks/{QUELLE}.py").read_text().split("\n") if l != "# Databricks notebook source"]
blocks, cur = [], []
for l in src + ["# COMMAND ----------"]:
    if l.strip() == "# COMMAND ----------":
        if any(x.strip() for x in cur):
            blocks.append(cur)
        cur = []
    else:
        cur.append(l)

cells = []
for b in blocks:
    while b and not b[0].strip():
        b = b[1:]
    while b and not b[-1].strip():
        b = b[:-1]
    if b[0].startswith("# MAGIC %md"):
        cells.append(("md", "\n".join(re.sub(r"^# MAGIC ?", "", l) for l in b[1:])))
    elif b[0].startswith("# MAGIC %pip"):
        cells.append(("code", re.sub(r"^# MAGIC ", "", b[0])))
    else:
        cells.append(("code", "\n".join(b)))

# Zelle "Code anlegen" direkt nach dem Neustart (Index 2) einfügen
parts = ["# Zelle 0: Legt den Projektcode (Ordner p229) auf dem Cluster an. Nichts ändern, nur ausführen.",
         "import os", 'os.makedirs("/tmp/p229_code/p229", exist_ok=True)', "FILES = {}"]
for n in FILES:
    code = (root / "p229" / f"{n}.py").read_text()
    assert "'''" not in code, n
    parts.append(f"FILES[{n!r}] = r'''{code}'''")
parts += ["for name, text in FILES.items():",
          '    open(f"/tmp/p229_code/p229/{name}.py", "w").write(text)',
          'print("Code angelegt:", sorted(os.listdir("/tmp/p229_code/p229")))']
cells.insert(3, ("code", "\n".join(parts)))

out = []
for kind, text in cells:
    if kind == "md":
        out.append({"cell_type": "markdown", "metadata": {}, "source": text})
    else:
        out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})
nb = {"cells": out, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(root / f"notebooks/{ZIEL}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("fertig,", len(out), "Zellen")
