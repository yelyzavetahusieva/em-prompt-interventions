import sys, time, nbformat
from nbclient import NotebookClient
nb = nbformat.read("notebooks/results.ipynb", as_version=4)
t = time.time()
NotebookClient(nb, timeout=1800, kernel_name="em-prompt-interventions", resources={"metadata": {"path": "notebooks"}}, allow_errors=True).execute()
nbformat.write(nb, "notebooks/results.ipynb")
bad = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        if o.output_type == "error":
            bad += 1
            print(f"--- cell {i} ERROR: {o.ename}: {o.evalue[:300]}")
            print("    first line of cell:", c.source.split("\n")[0][:110])
print(f"executed in {time.time()-t:.0f}s, errors: {bad}")
