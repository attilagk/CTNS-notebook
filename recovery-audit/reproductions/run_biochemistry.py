"""Fresh-kernel isolated execution for 2024-08-08-biochemistry."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NAME = "2024-08-08-biochemistry"
SOURCE = ROOT / NAME / f"{NAME}.ipynb"
RUN = Path(__file__).resolve().parent / "biochemistry-20261107T210813Z"
WORK = RUN / "work"
CWD = WORK / "notebooks" / NAME
BASE_PYTHON = Path("/home/attila/miniconda3/bin/python")
RESOURCE_DIR = ROOT.parent / "resources/QPS/5xFAD-mouse"
TARGETDOC = ROOT / "recovery-audit/target-artifacts/target-artifacts.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_layout() -> None:
    CWD.mkdir(parents=True, exist_ok=True)
    (WORK / "notebooks").mkdir(parents=True, exist_ok=True)
    (WORK / "results" / NAME).mkdir(parents=True, exist_ok=True)
    (RUN / "figures").mkdir(parents=True, exist_ok=True)
    for link, target in [
        (WORK / "notebooks/src", ROOT / "src"),
        (WORK / "resources", ROOT.parent / "resources"),
        (CWD / "named-figure", RUN / "figures"),
    ]:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"Refusing to replace non-symlink workspace path: {link}")
        link.symlink_to(target, target_is_directory=True)


def prepare_notebook(*, compatibility_fix: bool):
    nb = nbformat.read(SOURCE, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "%connect_info" in cell.source:
            cell.source = "# Omitted: prints live Jupyter connection credentials."
        cell.outputs = []
        cell.execution_count = None
    first = next(c for c in nb.cells if c.cell_type == "code")
    first.source = (
        "import sys\n"
        f"sys.path[:0] = [{str(ROOT / 'src')!r}, '/home/attila/src']\n"
        + first.source
    )
    if compatibility_fix:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(".applymap(", ".map(")
                cell.source = cell.source.replace(
                    "    data = Abeta_long.loc[b]\n"
                    "    data.loc[:, 'Treatment'] = pd.Categorical(data.Treatment, ordered=True, categories=treatments)",
                    "    data = Abeta_long.loc[b].copy()\n"
                    "    data = data.assign(Treatment=data.Treatment.cat.set_categories(treatments, ordered=True))",
                )
    nb.cells.append(nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        f"_run=Path({str(RUN)!r})\n"
        "assert sys.executable == '/home/attila/miniconda3/bin/python'\n"
        "results_params[('CO28154', 'TUDCA and HCQ')].to_csv(_run/'reproduced-target-table.csv')\n"
        "Abeta_long.to_csv(_run/'Abeta_long.csv', index=False)\n"
        "_ols=[]\n"
        "for _label, _frame in [('both_sexes', results), ('males_only', results_m)]:\n"
        "    for _formula in _frame.index:\n"
        "        for _experiment in _frame.columns:\n"
        "            _model=_frame.loc[_formula, _experiment]\n"
        "            _rank=int(np.linalg.matrix_rank(_model.model.exog))\n"
        "            _ols.append({'sex_subset':_label,'formula':_formula,'study':_experiment[0],'experiment':_experiment[1],'nobs':int(_model.nobs),'design_columns':int(_model.model.exog.shape[1]),'design_rank':_rank,'rank_deficient':_rank < _model.model.exog.shape[1]})\n"
        "(_run/'ols-design-diagnostics.json').write_text(json.dumps(_ols, indent=2)+'\\n')\n"
        "_nan=ttest_res[ttest_res.isna().any(axis=1)]\n"
        "_nan.to_csv(_run/'nonestimable-ttests.csv')\n"
    ))
    return nb


def execute(*, compatibility_fix: bool, output_name: str) -> dict:
    make_layout()
    n = prepare_notebook(compatibility_fix=compatibility_fix)
    out_path = RUN / output_name
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [str(BASE_PYTHON), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    events: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()
    error = None
    try:
        NotebookClient(
            n,
            km=km,
            timeout=1800,
            resources={"metadata": {"path": str(CWD)}},
            on_cell_start=lambda cell, cell_index, **kwargs: events.append(
                {"event": "cell_start", "cell": cell_index, "time": datetime.now(timezone.utc).isoformat()}
            ),
        ).execute()
        outcome = "completed"
    except Exception:
        outcome = "failed"
        error = traceback.format_exc()
    finally:
        nbformat.write(n, out_path)
        (RUN / (output_name.replace(".ipynb", "-events.json"))).write_text(json.dumps(events, indent=2) + "\n")
    return {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "compatibility_fix": compatibility_fix,
        "notebook": str(out_path),
        "error": error,
        "cells_started": [e["cell"] for e in events],
    }


def write_manifest() -> None:
    inputs = [
        RESOURCE_DIR / "CO28152-biochemistry-cleaned-Abeta.xlsx",
        RESOURCE_DIR / "CO28154-biochemistry-cleaned-Abeta.xlsx",
    ]
    helpers = [Path("/home/attila/src/attila_utils.py")]
    pip_packages = json.loads(subprocess.check_output([
        str(BASE_PYTHON), "-m", "pip", "list", "--format=json"
    ], text=True))
    conda_packages = json.loads(subprocess.check_output([
        "/home/attila/miniconda3/bin/conda", "list", "-n", "base", "--json"
    ], text=True))
    manifest = {
        "notebook": NAME,
        "conda_environment": "base",
        "python": str(BASE_PYTHON),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sha(SOURCE),
        "target_document_sha256": sha(TARGETDOC),
        "inputs": {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in inputs},
        "helpers": {str(p): sha(p) for p in helpers},
        "package_inventory_files": ["pip-packages.json", "conda-packages.json"],
        "dependency_check": {
            "required_imports": ["pandas", "numpy", "scipy", "matplotlib", "seaborn", "statsmodels", "statannotations", "attila_utils", "openpyxl"],
            "missing_dependencies_installed": {"statannotations": "0.7.2 from PyPI"},
            "omitted_debug_cell": "%connect_info (prints live kernel connection credentials)",
        },
        "criteria": {
            "table": "Compare all rows, columns, params, bse, and pvalues with target markdown at its displayed precision; treat displayed cells as rounded references.",
            "figure": "Compare the TUDCA + HCQ figure with the target reference for cohort/treatment/alloform/fraction/region mapping, log scale, annotations, labels, and visual pattern.",
        },
        "reproduction_compatibility_change": "pandas 3 removed DataFrame.applymap, and assigning a new categorical with fewer categories into an existing categorical column raises TypeError. Run copy changes summary calls to DataFrame.map and resets a copied plotting Treatment series with cat.set_categories(treatments, ordered=True). No model formula, input values, analysis selection, or inferential code changes.",
    }
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (RUN / "pip-packages.json").write_text(json.dumps(pip_packages, indent=2) + "\n")
    (RUN / "conda-packages.json").write_text(json.dumps(conda_packages, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / "mplconfig"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    RUN.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if mode != "verify":
        write_manifest()
    result = execute(compatibility_fix=(mode == "fixed"), output_name=f"{mode}-executed.ipynb")
    (RUN / f"{mode}-execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=2), flush=True)
    if result["error"]:
        (RUN / f"{mode}-error.txt").write_text(result["error"])
        print(result["error"], file=sys.stderr, flush=True)
    sys.exit(0 if result["outcome"] == "completed" else 1)
