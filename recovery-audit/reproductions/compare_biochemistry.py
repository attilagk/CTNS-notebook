"""Compare the 2024-08-08 biochemistry target table and figure reference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "recovery-audit/reproductions/biochemistry-20261107T210813Z"
TARGETDOC = ROOT / "recovery-audit/target-artifacts/target-artifacts.md"
TARGETIMAGE = ROOT / "recovery-audit/target-artifacts/Pasted image 20261106220452.png"
OUTPUTIMAGE = RUN / "figures/boxplot-Abeta-TUDCA-and-HCQ.png"
OUTPUTTABLE = RUN / "reproduced-target-table.csv"
INSPECTED_REFERENCE_SHA256 = "75d7f18d4a0e1432a1c3f28707412e09b3afe875dd757ca26b760f4770245f7a"
INSPECTED_OUTPUT_SHA256 = "0e6970952165474ec3f0ce00b87852ee4c697216f2bc3003d1439bd90bed7c4f"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tolerance_for(text: str) -> float:
    s = text.strip().replace(",", "")
    if "e" in s.lower():
        mantissa, exponent = s.lower().split("e")
        decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        return 0.5 * 10 ** (int(exponent) - decimals)
    decimals = len(s.split(".", 1)[1]) if "." in s else 0
    return 0.5 * 10 ** (-decimals)


def compare_table() -> dict:
    text = TARGETDOC.read_text()
    start = text.index("`results_params[('CO28154', 'TUDCA and HCQ')]`")
    start = text.index("```", start) + 3
    end = text.index("```", start)
    lines = [x.strip() for x in text[start:end].splitlines() if x.strip().startswith("|")]
    expected: dict[str, list[str]] = {}
    for line in lines:
        cells = [x.strip() for x in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] not in ("", "---") and not set(cells[0]) <= {"-", ":"}:
            try:
                [float(x) for x in cells[1:]]
            except ValueError:
                continue
            expected[cells[0]] = cells[1:]
    actual = pd.read_csv(OUTPUTTABLE, header=[0, 1], index_col=0)
    if list(actual.columns) != [
        ("both sexes", "params"), ("both sexes", "bse"), ("both sexes", "pvalues"),
        ("males only", "params"), ("males only", "bse"), ("males only", "pvalues"),
    ]:
        raise AssertionError(f"Unexpected table columns: {actual.columns.tolist()}")
    if set(actual.index) != set(expected):
        raise AssertionError(f"Target term mismatch: missing={set(expected)-set(actual.index)} extra={set(actual.index)-set(expected)}")
    comparisons = []
    for term, values in expected.items():
        for col, ref_text in zip(actual.columns, values):
            observed = float(actual.loc[term, col])
            reference = float(ref_text)
            delta = abs(observed - reference)
            tolerance = tolerance_for(ref_text)
            comparisons.append({
                "term": term,
                "dataset": col[0],
                "statistic": col[1],
                "reference": reference,
                "observed": observed,
                "absolute_difference": delta,
                "display_rounding_tolerance": tolerance,
                "passed": delta <= tolerance,
            })
    failures = [x for x in comparisons if not x["passed"]]
    return {
        "verdict": "passed" if not failures else "failed",
        "cells_compared": len(comparisons),
        "failure_count": len(failures),
        "maximum_absolute_difference": max(x["absolute_difference"] for x in comparisons),
        "criterion": "Each numeric value agrees with the markdown reference within half one unit of its last displayed digit.",
        "comparisons": comparisons,
    }


def compare_figure() -> dict:
    ref = Image.open(TARGETIMAGE).convert("RGB")
    out = Image.open(OUTPUTIMAGE).convert("RGB")
    ref_arr = np.asarray(ref, dtype=np.float64)
    resized = out.resize(ref.size, Image.Resampling.LANCZOS)
    out_arr = np.asarray(resized, dtype=np.float64)
    mae = float(np.abs(ref_arr - out_arr).mean())
    corr = float(np.corrcoef(ref_arr.reshape(-1), out_arr.reshape(-1))[0, 1])
    reference_sha = sha(TARGETIMAGE)
    output_sha = sha(OUTPUTIMAGE)
    inspected_pair_matches = (
        reference_sha == INSPECTED_REFERENCE_SHA256
        and output_sha == INSPECTED_OUTPUT_SHA256
    )
    return {
        "verdict": "visually matched for the inspected image hashes" if inspected_pair_matches else "uninspected: image hash differs from the inspected pair",
        "inspection_hashes_match": inspected_pair_matches,
        "reference": str(TARGETIMAGE.relative_to(ROOT)),
        "reproduced": str(OUTPUTIMAGE.relative_to(ROOT)),
        "reference_size_px": list(ref.size),
        "reproduced_size_px": list(out.size),
        "resized_mean_absolute_pixel_difference_0_255": mae,
        "resized_pixel_correlation": corr,
        "sha256": {"reference": reference_sha, "reproduced": output_sha},
        "inspected_features": [
            "2x2 panel order: FA cortex, FA hippocampus, DEA cortex, DEA hippocampus",
            "Saline TG and TUDCA + HCQ TG groups with matching colors and point distributions",
            "Aβ40/Aβ42 labels, log-scaled y axes, significance brackets/stars, legend and panel labels",
        ],
        "note": "Pixel scores are descriptive because the reference is a screenshot with different crop/resolution; visual acceptance is based on the inspected panels and data correspondence.",
    }


if __name__ == "__main__":
    result = {
        "notebook": "2024-08-08-biochemistry",
        "table": compare_table(),
        "figure": compare_figure(),
    }
    path = RUN / "target-comparisons.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"table": {k: v for k, v in result["table"].items() if k != "comparisons"}, "figure": result["figure"]}, indent=2, ensure_ascii=False))
    if result["table"]["verdict"] != "passed" or not result["figure"]["inspection_hashes_match"]:
        raise SystemExit(1)
