import sys, os, re, importlib
sys.path.insert(0, os.path.dirname(__file__))
import nbformat as nf
from nbformat.v4 import new_notebook, new_markdown_cell
from readings import READINGS
cells = []
for name in ("part1", "part2", "part3", "part4", "part5", "part6"):
    cells += importlib.import_module(name).CELLS
out_cells, used = [], set()
for c in cells:
    out_cells.append(c)
    if c.cell_type == "code":
        for name in re.findall(r'"([A-K]\d_[A-Za-z0-9_]+)"', c.source):
            if name in READINGS and name not in used:
                out_cells.append(new_markdown_cell("**Reading.** " + READINGS[name]))
                used.add(name)
                break
missing = sorted(set(READINGS) - used)
print("readings inserted:", len(used), "| unused readings:", missing)
nb = new_notebook(cells=out_cells, metadata={"kernelspec": {"name": "em-prompt-interventions", "display_name": "em-prompt-interventions", "language": "python"}})
out = "notebooks/results.ipynb"
nf.write(nb, out)
print("written", out, "cells:", len(out_cells))
