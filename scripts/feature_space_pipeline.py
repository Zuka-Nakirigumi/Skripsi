#!/usr/bin/env python3
"""Staged feature-space extractor for AID PyCIL (no training).

Stages: inspect-npz -> smoke (one production checkpoint per method) -> extract.
A project-specific adapter is REQUIRED for restoring models, loaders and features.
No fake embeddings, no unsafe automatic unpickling, no changes to metrics.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METHODS = ("fetril", "fecam", "ssre", "pass", "efc")
ATTENTIONS = ("none", "eca", "cbam", "procbam")
FEATURE_KEYS = ("features", "embeddings", "feature_vectors", "vectors")
AUDIT_FIELDS = [
    "run_id", "method", "protocol", "backbone", "attention", "seed", "batch", "machine",
    "checkpoint_path", "checkpoint_loadable", "feature_extractable", "feature_dim",
    "y_true_match", "sample_identity_match", "n_samples", "status", "error_message",
]


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_csv(p, rows, fields):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_run(run_dir, machine):
    p = Path(run_dir)
    args = read_json(p / "effective_args.json") if (p / "effective_args.json").is_file() else {}
    cfg = read_json(p / "config.json") if (p / "config.json").is_file() else {}
    name = p.name.lower()
    def find(name_, pattern, fallback=None):
        for doc in (args, cfg):
            if not isinstance(doc, dict):
                continue
            for key in (name_, name_ + "_name"):
                value = doc.get(key)
                if value is not None and not isinstance(value, (dict, list)):
                    return str(value).lower()
        m = re.search(pattern, name)
        return m.group(1) if m else fallback
    method = next((x for x in METHODS if re.search(fr"(^|_){x}(_|$)", name)), None)
    protocol = find("protocol", r"_(t(?:5|10))_", "")
    backbone = find("backbone", r"_(r(?:18|34))_", "")
    attention = find("attention", r"_(none|eca|cbam|procbam)_", "")
    method = str(args.get("method", cfg.get("method", method))).lower()
    seed = args.get("training_seed", args.get("seed", cfg.get("seed", 1993)))
    batch = args.get("batch_size", cfg.get("batch_size", ""))
    protocol = protocol.upper().replace("T_", "T")
    backbone = backbone.replace("resnet", "r").replace("_", "")
    if attention == "" or backbone == "" or protocol == "":
        # Some effective_args encode attention in convnet_type; run_id is the reliable fallback.
        m = re.search(r"_(none|eca|cbam|procbam)_seed", name)
        if m and not attention:
            attention = m.group(1)
        m = re.search(r"_(r18|r34)_", name)
        if m and not backbone:
            backbone = m.group(1)
        m = re.search(r"_(t5|t10)_", name)
        if m and not protocol:
            protocol = m.group(1).upper()
    n_task = 5 if protocol == "T5" else 10 if protocol == "T10" else None
    prediction_path = p / "predictions" / f"task_{n_task-1:02d}.npz" if n_task else None
    checkpoint_path = p / "model_checkpoints" / f"model_{n_task-1}.pkl" if n_task else None
    return {
        "run_id": p.name, "run_dir": p, "machine": machine, "method": method,
        "protocol": protocol, "backbone": backbone, "attention": attention,
        "seed": seed, "batch": batch, "n_task": n_task,
        "prediction_path": prediction_path, "checkpoint_path": checkpoint_path,
        "effective_args": args,
    }


def scan_runs(root, pc_source, server_source):
    sources = {"pc2": root / pc_source, "server": root / server_source}
    runs = []
    for machine, folder in sources.items():
        if not folder.is_dir():
            continue
        for d in sorted(folder.iterdir()):
            if d.is_dir() and (d / "effective_args.json").is_file():
                # Never treat incomplete or sanity/profiling runs as production.
                if not (d / "status.json").is_file():
                    continue
                try:
                    if read_json(d / "status.json").get("state") != "completed":
                        continue
                except Exception:
                    continue
                run = parse_run(d, machine)
                if run["method"] in METHODS and run["protocol"] in ("T5", "T10"):
                    runs.append(run)
    return runs


def npz_info(run):
    p = run["prediction_path"]
    rec = {k: str(run[k]) for k in ("run_id", "machine", "method", "protocol", "backbone", "attention")}
    rec["prediction_path"] = str(p)
    if p is None or not p.is_file():
        return {**rec, "status": "missing_predictions", "keys": "", "has_embeddings": False, "arrays": "{}"}
    try:
        with np.load(p, allow_pickle=False) as ar:
            infos = {}
            for key in ar.files:
                a = ar[key]
                infos[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            found = [k for k in FEATURE_KEYS if k in ar.files]
            has = bool(found)
            if has:
                shape = ar[found[0]].shape
                has = len(shape) == 2 and shape[0] > 0
            return {**rec, "status": "ok", "keys": ",".join(ar.files),
                    "has_embeddings": has, "arrays": json.dumps(infos, ensure_ascii=False)}
    except Exception as exc:
        return {**rec, "status": "unreadable: " + str(exc), "keys": "", "has_embeddings": False, "arrays": "{}"}


def reference_predictions(run):
    p = run["prediction_path"]
    if p is None or not p.is_file():
        raise FileNotFoundError(f"Missing canonical predictions: {p}")
    with np.load(p, allow_pickle=False) as f:
        if not {"y_true", "y_pred_topk"}.issubset(f.files):
            raise ValueError(f"Missing y_true/y_pred_topk in {p}: {f.files}")
        y_true = np.asarray(f["y_true"], dtype=np.int64)
        pred = np.asarray(f["y_pred_topk"], dtype=np.int64)
        pred = pred[:, 0] if pred.ndim == 2 else pred
        ids = np.asarray(f["sample_id"]).astype(str) if "sample_id" in f.files else None
    if y_true.ndim != 1 or pred.ndim != 1 or y_true.shape != pred.shape:
        raise ValueError("Invalid canonical prediction shapes")
    return y_true, pred, ids


def load_adapter(filepath):
    if not filepath:
        raise RuntimeError("--adapter is required. See scripts/feature_adapter_template.py")
    p = Path(filepath).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("pycil_feature_adapter", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "extract_from_checkpoint", None)):
        raise RuntimeError("Adapter must define extract_from_checkpoint(...) and return a dict")
    return mod


def verify_extraction(result, run, full, require_2000=True):
    if not isinstance(result, dict):
        raise TypeError("Adapter must return a dictionary")
    if result.get("checkpoint_loaded") is not True:
        raise ValueError("Adapter did not confirm restoring checkpoint weights")
    if result.get("semantic_eval_features") is not True:
        raise ValueError("Adapter did not confirm pre-classifier semantic eval embeddings")
    x = np.asarray(result.get("features"), dtype=np.float32)
    y = np.asarray(result.get("y_true"), dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[1] <= 0:
        raise ValueError(f"Invalid feature dimensions: features={x.shape}, y_true={y.shape}")
    if not np.isfinite(x).all() or np.linalg.norm(x, axis=1).min() < 1e-12:
        raise ValueError("Features have NaN/Inf or zero-norm samples")
    if x.shape[1] != 512:
        raise ValueError(f"Expected documented 512-d feature representation, got {x.shape[1]}.")
    ref_y, ref_pred, ref_ids = reference_predictions(run)
    if full:
        if require_2000 and len(y) != 2000:
            raise ValueError(f"AID final task expected 2000 test samples, received {len(y)}")
        if len(y) != len(ref_y) or not np.array_equal(y, ref_y):
            raise ValueError("Full y_true does not match canonical artifact in exact order")
    else:
        if len(y) > len(ref_y) or not np.array_equal(y, ref_y[:len(y)]):
            raise ValueError("Smoke y_true does not match prefix of canonical prediction artifact")
    raw_ids = result.get("sample_id")
    ids = None if raw_ids is None else np.asarray(raw_ids).astype(str)
    raw_paths = result.get("sample_path")
    paths = None if raw_paths is None else np.asarray(raw_paths).astype(str)
    if ids is not None and (len(ids) != len(y) or len(set(ids.tolist())) != len(ids)):
        raise ValueError("sample_id is invalid or contains duplicates")
    if paths is not None and len(paths) != len(y):
        raise ValueError("sample_path length mismatch")
    sample_verification = "label_order_only"
    if ids is not None and ref_ids is not None:
        if not np.array_equal(ids, ref_ids if full else ref_ids[:len(y)]):
            raise ValueError("sample_id order does not match canonical prediction artifact")
        sample_verification = "sample_ids_match"
    return x, y, (ref_pred if full else ref_pred[:len(y)]), ids, paths, sample_verification


def make_audit(run, status="missing", **kwargs):
    row = {k: run.get(k, "") for k in ("run_id", "method", "protocol", "backbone", "attention", "seed", "batch", "machine")}
    row.update({"checkpoint_path": str(run.get("checkpoint_path") or ""),
                "checkpoint_loadable": False, "feature_extractable": False, "feature_dim": "",
                "y_true_match": False, "sample_identity_match": "unknown", "n_samples": "",
                "status": status, "error_message": ""})
    row.update(kwargs)
    return row


def select_smoke(runs):
    selected = []
    for method in METHODS:
        choices = [r for r in runs if r["method"] == method]
        if not choices:
            selected.append(None)
            continue
        choices.sort(key=lambda r: (
            r["attention"] != "none", r["protocol"] != "T5", r["backbone"] != "r18", r["run_id"]
        ))
        selected.append(choices[0])
    return selected


def extract_one(adapter, run, limit, output_root, full):
    cp = run["checkpoint_path"]
    if cp is None or not cp.is_file():
        raise FileNotFoundError(f"Checkpoint missing on THIS machine: {cp}. Snapshot inventories are insufficient.")
    # Adapter owns PyCIL construction and state_dict loading; never guess learner implementation.
    result = adapter.extract_from_checkpoint(
        run_dir=run["run_dir"], checkpoint_path=cp,
        effective_args=run["effective_args"], method=run["method"],
        protocol=run["protocol"], backbone=run["backbone"], attention=run["attention"],
        limit=limit,
    )
    x, y, pred, ids, paths, level = verify_extraction(result, run, full=full)
    if full:
        dest = output_root / "feature_embeddings" / f"{run['run_id']}_final.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(features=x, y_true=y, y_pred=pred,
                      task_id=np.int32(run["n_task"]-1), run_id=np.str_(run["run_id"]))
        if ids is not None:
            arrays["sample_id"] = ids
        if paths is not None:
            arrays["sample_path"] = paths
        tmp = dest.with_name(dest.name + ".partial")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(dest)
    return make_audit(run, status="passed", checkpoint_loadable=True, feature_extractable=True,
                      feature_dim=x.shape[1], y_true_match=True, n_samples=len(y),
                      sample_identity_match=level)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["inspect-npz", "smoke", "extract"])
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--pc-source", default="outputs/experiments")
    ap.add_argument("--server-source", default="analysis_inputs/server/experiments")
    ap.add_argument("--adapter", default=None, help="Absolute or project-relative local adapter implementation")
    ap.add_argument("--output-dir", default="outputs/feature_analysis")
    ap.add_argument("--mode", choices=["baseline", "all"], default="baseline")
    ap.add_argument("--machine", choices=["all", "pc2", "server"], default="all",
                    help="Only use checkpoint files from this machine; necessary for distributed production")
    ap.add_argument("--smoke-samples", type=int, default=16)
    ap.add_argument("--global-smoke-summary", default=None,
                    help="Optional merged five-method smoke JSON from scripts/merge_feature_smoke.py")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    runs = scan_runs(root, args.pc_source, args.server_source)
    if args.machine != "all":
        runs = [r for r in runs if r["machine"] == args.machine]
    if args.stage == "inspect-npz":
        rows = [npz_info(run) for run in runs]
        save_csv(output / "prediction_npz_audit.csv", rows,
                 ["run_id", "machine", "method", "protocol", "backbone", "attention",
                  "prediction_path", "status", "keys", "has_embeddings", "arrays"])
        print(json.dumps({"runs_scanned": len(rows),
                          "npz_with_embeddings": sum(int(r["has_embeddings"]) for r in rows),
                          "output": str(output / "prediction_npz_audit.csv")}, indent=2))
        return
    adapter = load_adapter(root / args.adapter if args.adapter and not Path(args.adapter).is_absolute() else args.adapter)
    if args.stage == "smoke":
        audit = []
        for method, run in zip(METHODS, select_smoke(runs)):
            if run is None:
                audit.append(make_audit({"method": method}, status="no_local_run", error_message="No matching production run"))
                continue
            if run["checkpoint_path"] is None or not run["checkpoint_path"].is_file():
                local = [r for r in runs if r["method"] == method and r["checkpoint_path"] is not None
                         and r["checkpoint_path"].is_file()]
                if not local:
                    audit.append(make_audit(run, status="no_local_checkpoint", error_message="Only snapshot/remote metadata exists"))
                    continue
                local.sort(key=lambda r: (r["attention"] != "none", r["protocol"] != "T5", r["backbone"] != "r18"))
                run = local[0]
            try:
                audit.append(extract_one(adapter, run, args.smoke_samples, output, full=False))
            except Exception as exc:
                audit.append(make_audit(run, status="failed", error_message=f"{type(exc).__name__}: {exc}"))
            print(method, audit[-1]["status"], audit[-1]["error_message"])
        save_csv(output / "feature_checkpoint_audit.csv", audit, AUDIT_FIELDS)
        write_json(output / "feature_smoke_summary.json", {
            "at": datetime.now(timezone.utc).isoformat(), "per_method": audit,
            "ready_for_batch": len(audit) == 5 and all(r["status"] == "passed" for r in audit),
            "note": "Only one production final-task run per method; not result data."})
        print("Smoke audit saved:", output / "feature_checkpoint_audit.csv")
        return
    summary_file = output / "feature_smoke_summary.json"
    if not summary_file.exists():
        raise RuntimeError("Missing smoke summary. Run 'smoke' first. Batch extraction is blocked.")
    sm = read_json(summary_file)
    global_passed = False
    if args.global_smoke_summary:
        gp = Path(args.global_smoke_summary)
        if not gp.is_absolute():
            gp = root / gp
        global_passed = bool(read_json(gp).get("ready_for_batch"))
    if not sm.get("ready_for_batch") and not global_passed:
        raise RuntimeError("Five-method production smoke audit not passed globally. Batch extraction blocked.")
    # Local methods must independently pass on the machine actually performing inference.
    local_passed = {r.get("method") for r in sm.get("per_method", []) if r.get("status") == "passed"}
    selected = [r for r in runs if (args.mode == "all" or r["attention"] == "none")
                and r["checkpoint_path"] is not None and r["checkpoint_path"].is_file()]
    missing_local = {r["method"] for r in selected} - local_passed
    if missing_local:
        raise RuntimeError(f"Local checkpoint adapter smoke did not pass for methods {sorted(missing_local)}")
    audit = []
    for i, run in enumerate(selected, 1):
        try:
            row = extract_one(adapter, run, limit=None, output_root=output, full=True)
        except Exception as exc:
            row = make_audit(run, status="failed", error_message=f"{type(exc).__name__}: {exc}")
        audit.append(row)
        print(f"[{i}/{len(selected)}] {run['run_id']}: {row['status']} {row['error_message']}", flush=True)
    save_csv(output / "feature_extraction_audit.csv", audit, AUDIT_FIELDS)
    print("Extracted:", sum(r["status"] == "passed" for r in audit), "/", len(selected))

if __name__ == "__main__":
    main()
