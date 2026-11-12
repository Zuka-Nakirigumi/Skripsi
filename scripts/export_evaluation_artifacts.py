#!/usr/bin/env python3
"""Create a portable metrics-only ZIP for 07_results_analysis.ipynb.

Run from PyCIL root on each source computer, using the original Python env.
Example:
  python scripts/export_evaluation_artifacts.py --machine server

The ZIP stores JSON, canonical prediction NPZ and confusion matrix NPY for
completed runs, checkpoint manifests, and a *structural inventory* of model
and method-state .pkl files (but NOT actual model weights).

This intentionally does not call torch.load or pickle.load; checkpoint
inventory verifies source presence and non-zero length ONLY.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_JSON = (
    "status.json", "config.json", "effective_args.json", "provenance.json",
    "summary.json", "task_metrics.json", "continual_metrics.json",
)


def inventory(run_dir: Path) -> dict:
    ckpt = run_dir / "model_checkpoints"
    files = sorted(ckpt.glob("*.pkl")) if ckpt.is_dir() else []
    results = {}
    for item in files:
        if not (item.name.startswith("model_") or item.name.startswith("method_state_")):
            continue
        rel = "model_checkpoints/" + item.name
        results[rel] = {
            "exists": item.is_file(),
            "size": int(item.stat().st_size) if item.is_file() else 0,
        }
    return {
        "source_machine": ARGS.machine,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Presence and file size checked on source only; content/loadability not verified.",
        "inventory": results,
    }


def included_files(run_dir: Path):
    for file in MAIN_JSON:
        path = run_dir / file
        if path.is_file():
            yield path
    final = run_dir / "final_confusion_matrix.npy"
    if final.is_file(): yield final
    for subdir, pattern in (("predictions", "task_*.npz"),
                            ("confusion_matrices", "task_*.npy"),
                            ("model_checkpoints", "checkpoint_*.json")):
        folder = run_dir / subdir
        if folder.is_dir():
            yield from sorted(p for p in folder.glob(pattern) if p.is_file())


def run_status(run_dir: Path) -> str:
    p = run_dir / "status.json"
    if not p.exists(): return "missing"
    try:
        return str(json.loads(p.read_text(encoding="utf-8"))["state"])
    except Exception:
        return "unreadable"


def main():
    src = (PROJECT_ROOT / ARGS.source).resolve()
    outroot = (PROJECT_ROOT / ARGS.output_dir).resolve()
    if not src.is_dir():
        raise SystemExit(f"No source directory: {src}")
    outroot.mkdir(parents=True, exist_ok=True)
    target = outroot / f"{ARGS.machine}_evaluation_{datetime.now():%Y%m%d_%H%M%S}.zip"
    temp = target.with_suffix(".zip.part")
    n_runs = n_files = n_ignored = 0
    manifest = []
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3,
                             allowZip64=True) as zf:
            for run_dir in sorted(x for x in src.iterdir() if x.is_dir()):
                state = run_status(run_dir)
                if state == "missing":
                    continue
                if state != "completed" and not ARGS.include_incomplete:
                    n_ignored += 1
                    continue
                n_runs += 1
                source_files = list(included_files(run_dir))
                for path in source_files:
                    zf.write(path, f"experiments/{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
                    n_files += 1
                # Always write inventory, including empty inventory; notebook will reject incomplete checkpoints.
                zf.writestr(f"experiments/{run_dir.name}/checkpoint_inventory.json",
                            json.dumps(inventory(run_dir), indent=2))
                manifest.append({"run_id": run_dir.name, "state_at_export": state,
                                 "file_count": len(source_files)})
            zf.writestr("export_info.json", json.dumps({
                "source_machine": ARGS.machine,
                "export_time": datetime.now().isoformat(timespec="seconds"),
                "source_directory": str(src),
                "include_incomplete": ARGS.include_incomplete,
                "runs": manifest,
                "notes": "No model weights included. Checkpoint inventory is not torch.load verification."
            }, indent=2))
        os.replace(temp, target)
    except Exception:
        if temp.exists(): temp.unlink()
        raise
    print(f"Export created: {target}")
    print(f"Runs: {n_runs}; files: {n_files}; noncompleted omitted: {n_ignored}")
    print(f"Size: {target.stat().st_size / 1024**2:.1f} MiB")
    print("Extract ZIP into PyCIL/analysis_inputs/<machine>/ (not directly into experiments/).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=("server", "pc2"), required=True)
    parser.add_argument("--source", default="outputs/experiments")
    parser.add_argument("--output-dir", default="outputs/eval_exports")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Export incomplete/failed records too; avoid while jobs are actively writing.")
    ARGS = parser.parse_args()
    main()
