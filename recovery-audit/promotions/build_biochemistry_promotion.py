"""Create the canonical notebook from the validated full-kernel reproduction."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "recovery-audit/reproductions/biochemistry-20261107T210813Z"
SOURCE = RUN / "fixed-executed.ipynb"
DESTINATION = ROOT / "2024-08-08-biochemistry/2024-08-08-biochemistry.ipynb"

nb = nbformat.read(SOURCE, as_version=4)

# The final harness cell exports comparison data only for the run record.
nb.cells = [
    cell for cell in nb.cells
    if not (
        cell.cell_type == "code"
        and "results_params[('CO28154', 'TUDCA and HCQ')].to_csv(_run/'reproduced-target-table.csv')" in cell.source
    )
]

# The reproduction supplied these paths externally; keep that bootstrap out of
# the canonical notebook so it continues using the user's ordinary IPython setup.
first_code = next(cell for cell in nb.cells if cell.cell_type == "code")
prefix = "import sys\nsys.path[:0] = ["
if first_code.source.startswith(prefix):
    first_code.source = first_code.source.split("\n", 2)[2]

# Never retain the Jupyter connection-information magic, even as a commented
# execution cell. It prints live kernel credentials and has no scientific role.
nb.cells = [
    cell for cell in nb.cells
    if not (cell.cell_type == "code" and "prints live Jupyter connection credentials" in cell.source)
]

nbformat.validate(nb)
nbformat.write(nb, DESTINATION)
print(f"Wrote {DESTINATION} with {len(nb.cells)} cells")
