#!/usr/bin/env python3
"""Read-only checkpoint and prediction inventory for PyCIL AID production.

Does NOT call torch.load, unpickle checkpoint files, construct models or start training.
Generates CSV inventories and a small JSON summary to plan one smoke checkpoint per method.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUN_RE = re.compile(
    r"^aid_(fetril|fecam|ssre|pass|efc)_(t5|t10)_(r18|r34)_(none|eca|cbam|procbam)_seed(\d+)$",
    re.IGNORECASE,
)
METHODS = ("fetril", "fecam", "ssre", "pass", "efc")
COLUMNS = (
    "run_id", "method", "protocol", "backbone", "attention", "seed", "machine",
    "status_state", "final_task", "final_checkpoint_exists", "final_checkpoint_bytes",
    "final_method_state_exists", "final_method_state_bytes", "final_manifest_exists",
    "final_checkpoint_relative_path", "prediction_exists", "prediction_keys", "y_true_shape",
    "y_true_dtype", "y_true_count", "y_pred_topk_shape", "sample_id_present",
    "checkpoint_manifest_keys", "preflight_status", "notes",
)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def safe_stats(path):
    try:
        return path.is_file() and path.stat().st_size > 0, path.stat().st_size if path.is_file() else 0
    except OSError:
        return False, 0


def write_csv(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inspect_run(run, machine):
    m = RUN_RE.match(run.name)
    if not m:
        return None
    method, proto, backbone, attention, seed = (v.lower() for v in m.groups())
    task_id = 4 if proto == "t5" else 9
    ckpt = run / "model_checkpoints" / f"model_{task_id}.pkl"
    state = run / "model_checkpoints" / f"method_state_{task_id}.pkl"
    manifest = run / "model_checkpoints" / f"checkpoint_{task_id}.json"
    prediction = run / "predictions" / f"task_{task_id:02d}.npz"
    status = read_json(run / "status.json") or {}
    current_state = status.get("state", "unknown") if isinstance(status, dict) else "unknown"
    ckpt_exists, ckpt_bytes = safe_stats(ckpt)
    state_exists, state_bytes = safe_stats(state)
    manifest_exists, _ = safe_stats(manifest)
    pred_exists, _ = safe_stats(prediction)
    manifest_keys = ""
    info = read_json(manifest) if manifest_exists else None
    if isinstance(info, dict):
        manifest_keys = ",".join(sorted(info.keys()))
    pred_keys = ""
    y_true_shape = ""
    y_pred_shape = ""
    y_true_dtype = ""
    y_true_count = ""
    sample_id_present = False
    issues = []
    if pred_exists:
        try:
            with np.load(prediction, allow_pickle=False) as arrays:
                pred_keys = ",".join(arrays.files)
                if "y_true" in arrays.files:
                    yt = arrays["y_true"]
                    y_true_shape = str(tuple(yt.shape))
                    y_true_dtype = str(yt.dtype)
                    y_true_count = len(yt) if yt.ndim else 0
                else:
                    issues.append("missing_y_true")
                if "y_pred_topk" in arrays.files:
                    y_pred_shape = str(tuple(arrays["y_pred_topk"].shape))
                else:
                    issues.append("missing_y_pred_topk")
                sample_id_present = "sample_id" in arrays.files
        except Exception as exc:
            issues.append(f"predictions_unreadable:{type(exc).__name__}")
    else:
        issues.append("missing_final_predictions")
    if current_state != "completed":
        issues.append("status_not_completed")
    if not ckpt_exists:
        issues.append("missing_final_model_checkpoint")
    if not state_exists:
        issues.append("missing_final_method_state")
    if not manifest_exists:
        issues.append("missing_checkpoint_manifest")
    if y_true_count not in ("", 2000):
        issues.append("unexpected_final_test_count")
    return {
        "run_id": run.name, "method": method, "protocol": proto.upper(),
        "backbone": backbone, "attention": attention, "seed": int(seed),
        "machine": machine, "status_state": current_state, "final_task": task_id,
        "final_checkpoint_exists": ckpt_exists, "final_checkpoint_bytes": ckpt_bytes,
        "final_method_state_exists": state_exists, "final_method_state_bytes": state_bytes,
        "final_manifest_exists": manifest_exists,
        "final_checkpoint_relative_path": f"{run.name}/model_checkpoints/model_{task_id}.pkl",
        "prediction_exists": pred_exists, "prediction_keys": pred_keys,
        "y_true_shape": y_true_shape, "y_true_dtype": y_true_dtype,
        "y_true_count": y_true_count, "y_pred_topk_shape": y_pred_shape,
        "sample_id_present": sample_id_present, "checkpoint_manifest_keys": manifest_keys,
        "preflight_status": "inventory_ready" if not issues else "review_required",
        "notes": ";".join(issues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, choices=("pc2", "server"))
    parser.add_argument("--source", default="outputs/experiments")
    parser.add_argument("--output-dir", default="outputs/feature_analysis")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        parser.error(f"Source directory not found: {source}")

    rows = [r for d in sorted(source.iterdir()) if d.is_dir() for r in [inspect_run(d, args.machine)] if r]
    write_csv(output / "checkpoint_preflight_inventory.csv", rows, COLUMNS)
    smoke = []
    for method in METHODS:
        candidates = [r for r in rows if r["method"] == method and r["final_checkpoint_exists"]
                      and r["preflight_status"] == "inventory_ready"]
        candidates.sort(key=lambda r: (r["attention"] != "none", r["protocol"] != "T5",
                                       r["backbone"] != "r18", r["run_id"]))
        if candidates:
            chosen = dict(candidates[0])
            chosen["candidate_status"] = "selected_for_smoke"
        else:
            chosen = {"method": method, "machine": args.machine, "candidate_status": "no_local_ready_checkpoint"}
        smoke.append(chosen)
    write_csv(output / "checkpoint_smoke_candidates.csv", smoke, list(COLUMNS) + ["candidate_status"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": args.machine,
        "runs_scanned": len(rows),
        "inventory_ready": sum(r["preflight_status"] == "inventory_ready" for r in rows),
        "checkpoint_present": sum(bool(r["final_checkpoint_exists"]) for r in rows),
        "by_method": dict(sorted(Counter(r["method"] for r in rows).items())),
        "smoke_candidates": [{k: v for k, v in r.items() if k in (
            "method", "run_id", "backbone", "protocol", "attention", "candidate_status", "final_checkpoint_relative_path"
        )} for r in smoke],
        "files": ["checkpoint_preflight_inventory.csv", "checkpoint_smoke_candidates.csv"],
        "important": "Inventory only; no checkpoint was loaded. Requires learner-specific adapter before feature extraction.",
    }
    (output / "checkpoint_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("machine", "runs_scanned", "inventory_ready", "checkpoint_present", "by_method", "smoke_candidates")},
                     indent=2, ensure_ascii=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
