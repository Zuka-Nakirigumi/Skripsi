"""Targeted production recovery for PyCIL sections.

Copy to PyCIL/scripts/retry_incomplete_section.py and run from PyCIL root.

This script imports the CURRENT production_section.py, so PASS section
registries and all original run configurations are reused, not redefined.
It reuses the original section's activity-log folder, preserves previous logs,
backs up/merges the section report, and archives incomplete per-run output
before restarting a full incremental run. It does not implement mid-task resume.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for directory in (str(ROOT), str(SCRIPT_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

# Use the active Python section registry, canonical config and verifier.
from production_section import (  # noqa: E402
    ATTENTIONS,
    build_run_config,
    preflight_canonical_cache,
    run_experiment,
    section_table,
    verify_run_output,
)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, path)


def get_session_dir(machine: str, section: int, explicit: str | None) -> Path:
    activity_root = ROOT / "outputs" / "activity_logs"
    prefix = f"{machine}_section{section:02d}_"
    if explicit:
        candidate = Path(explicit)
        candidate = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    else:
        choices = sorted(p for p in activity_root.glob(prefix + "*") if p.is_dir())
        if not choices:
            raise FileNotFoundError(
                f"Tidak ditemukan folder activity log {prefix}* di {activity_root}. "
                "Tentukan --session-dir secara eksplisit."
            )
        candidate = choices[-1].resolve()
    if not candidate.is_dir() or not candidate.name.startswith(prefix):
        raise ValueError(
            f"Folder {candidate} bukan direktori log {machine} Section {section}."
        )
    if not candidate.is_relative_to(activity_root.resolve()):
        raise ValueError("--session-dir harus berada di outputs/activity_logs.")
    return candidate


def resolve_root(value: str) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def build_spec(spec: dict, attention: str, seed: int) -> dict:
    return build_run_config(
        dataset=spec["dataset"],
        method=spec["method"],
        protocol=spec["protocol"],
        backbone=spec["backbone"],
        attention=attention,
        training_seed=seed,
        method_overrides={"batch_size": int(spec["batch_size"])},
    )


def inspect_run(run_dir: Path, protocol: str) -> tuple[bool, dict | str]:
    if not run_dir.exists():
        return False, "No existing run folder"
    try:
        result = verify_run_output(run_dir, protocol=protocol)
        return True, result
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def archive_incomplete(run_dir: Path, machine: str, section: int, attempt: str) -> Path | None:
    if not run_dir.exists():
        return None
    archive_root = ROOT / "outputs" / "recovery_archives" / f"{machine}_section{section:02d}"
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / f"{run_dir.name}__{attempt}"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {dest}")
    # Move, never delete. Same-volume rename is atomic; shutil.move also handles mounts.
    shutil.move(str(run_dir), str(dest))
    return dest


class DualLog:
    def __init__(self, original, *logfiles):
        self.original = original
        self.logfiles = logfiles
        self.lock = threading.RLock()

    def write(self, value):
        with self.lock:
            self.original.write(value)
            for handle in self.logfiles:
                handle.write(value)
        return len(value)

    def flush(self):
        with self.lock:
            self.original.flush()
            for handle in self.logfiles:
                handle.flush()

    def isatty(self):
        return bool(getattr(self.original, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def fileno(self):
        return self.original.fileno()


def command_snapshot(args: list[str], timeout: int = 60) -> str:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, errors="replace")
        return (result.stdout + result.stderr).strip() or f"[exit {result.returncode}, no output]"
    except Exception as exc:
        return f"[{type(exc).__name__}: {exc}]"


def write_retry_snapshots(session: Path, attempt: str, args: argparse.Namespace) -> None:
    lines = [
        f"Retry timestamp: {iso_now()}",
        f"Machine: {args.machine}",
        f"Section: {args.section}",
        f"Attentions: {', '.join(args.attentions)}",
        f"Python: {sys.executable}",
        f"Project: {ROOT}",
        f"Device: {args.device}",
        f"Workers: {args.num_workers}",
        "--- git rev-parse HEAD ---",
        command_snapshot(["git", "rev-parse", "HEAD"]),
        "--- git status --short ---",
        command_snapshot(["git", "status", "--short"]),
        "--- python --version ---",
        command_snapshot([sys.executable, "--version"]),
        "--- nvidia-smi initial ---",
        command_snapshot(["nvidia-smi"]),
    ]
    (session / f"retry_environment_{attempt}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (session / f"retry_pip_freeze_{attempt}.txt").write_text(
        command_snapshot([sys.executable, "-m", "pip", "freeze"], timeout=120) + "\n", encoding="utf-8"
    )


def gpu_monitor(session: Path, device: int, stop: threading.Event) -> None:
    csv_path = session / "gpu_usage.csv"
    header = (
        "timestamp,index,name,gpu_util_percent,memory_util_percent,"
        "memory_used_mib,memory_total_mib,temperature_c,power_draw_w,power_limit_w\n"
    )
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("a", encoding="utf-8") as handle:
            handle.write(header)
    query = (
        "timestamp,index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    )
    while not stop.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(device), "--query-gpu=" + query,
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8, errors="replace"
            )
            if result.returncode == 0 and result.stdout.strip():
                with csv_path.open("a", encoding="utf-8") as handle:
                    handle.write(result.stdout.rstrip("\n") + "\n")
        except Exception:
            pass
        stop.wait(10)


def load_report(session: Path, machine: str, section: int, spec: dict, args: argparse.Namespace) -> dict:
    report_path = session / "section_report.json"
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if old.get("machine") not in (None, machine) or int(old.get("section", section)) != section:
            raise ValueError("section_report.json does not match requested machine/section")
        backup = session / f"section_report.before_retry_{args.attempt}.json"
        shutil.copy2(report_path, backup)
        report = old
    else:
        report = {
            "machine": machine, "section": section, "label": spec["label"],
            "method": spec["method"], "dataset": spec["dataset"],
            "protocol": spec["protocol"], "backbone": spec["backbone"],
            "batch_size": int(spec["batch_size"]), "seed": args.seed,
            "attentions": list(ATTENTIONS), "results": [], "started_at": iso_now(),
        }
    report.setdefault("results", [])
    report.setdefault("retry_history", [])
    return report


def upsert_result(report: dict, attention: str, values: dict) -> None:
    records = report.setdefault("results", [])
    matches = [i for i, item in enumerate(records) if item.get("attention") == attention]
    if matches:
        records[matches[-1]].update(values)
    else:
        records.append({"attention": attention, **values})


def execute(args: argparse.Namespace) -> int:
    spec = section_table(args.machine)[args.section]
    if spec["method"] != "pass":
        raise ValueError("Safety check: this recovery script is intended for PASS sections only")
    session = get_session_dir(args.machine, args.section, args.session_dir)
    output_root = resolve_root(args.output_root)
    resolved_runs = {a: build_spec(spec, a, args.seed) for a in args.attentions}
    print(f"SECTION: {args.machine.upper()} {args.section} | {spec['label']}")
    print(f"SESSION: {session}")
    print(f"EXPERIMENTS: {output_root}")
    print(f"GPU: {args.device} | batch: {spec['batch_size']} | workers: {args.num_workers} | seed: {args.seed}")
    for attention, config in resolved_runs.items():
        target = output_root / config["run_id"]
        is_good, detail = inspect_run(target, spec["protocol"])
        print(f"  {attention:8s} -> {config['run_id']} | {'VERIFIED/SKIP' if is_good else 'RETRAIN'}")
        if not is_good:
            print(f"      Why: {detail}")
    if args.dry_run:
        print("DRY RUN ONLY: no training, files, reports or logs have been changed.")
        return 0

    # Preflight before logging/recovery changes. Errors here leave previous records untouched.
    preflight_canonical_cache(spec["method"])
    args.attempt = stamp()
    section_log = (session / "section.log").open("a", encoding="utf-8", buffering=1)
    retry_log = (session / f"retry_{args.attempt}.log").open("a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = DualLog(old_out, section_log, retry_log)
    sys.stderr = DualLog(old_err, section_log, retry_log)
    stop = threading.Event()
    monitor_thread = None
    report = None
    report_path = session / "section_report.json"
    try:
        print(f"\n{'=' * 72}\nRETRY BEGIN {iso_now()}: section {args.section}, {args.attentions}\n{'=' * 72}", flush=True)
        write_retry_snapshots(session, args.attempt, args)
        report = load_report(session, args.machine, args.section, spec, args)
        report["retry_history"].append({
            "timestamp": iso_now(), "attempt_id": args.attempt,
            "attentions": list(args.attentions), "state": "running",
            "retry_log": f"retry_{args.attempt}.log",
        })
        report["state"] = "running"
        json_write_atomic(report_path, report)
        monitor_thread = threading.Thread(target=gpu_monitor, args=(session, args.device, stop), daemon=True)
        monitor_thread.start()

        for attention, config in resolved_runs.items():
            run_id = config["run_id"]
            run_dir = output_root / run_id
            good, detail = inspect_run(run_dir, spec["protocol"])
            if good:
                print(f"[SKIP verified] {attention} | {run_id}", flush=True)
                upsert_result(report, attention, {**detail, "run_id": run_id,
                                                  "state": "verified", "verified_at": iso_now()})
                json_write_atomic(report_path, report)
                continue
            print(f"[RETRAIN] {attention} | {run_id}; prior state: {detail}", flush=True)
            archived = archive_incomplete(run_dir, args.machine, args.section, args.attempt)
            if archived:
                print(f"[ARCHIVED prior incomplete output] {archived}", flush=True)
            entry = {"run_id": run_id, "attention": attention, "state": "running",
                     "started_at": iso_now(), "retry_attempt": args.attempt}
            if archived:
                entry["archived_prior_run"] = str(archived)
            upsert_result(report, attention, entry)
            json_write_atomic(report_path, report)

            # IMPORTANT: full T5/T10 run, same run_id and same output-root as production.
            summary = run_experiment(
                config, output_root=str(output_root), device_index=args.device,
                num_workers=args.num_workers, max_tasks=None, run_mode="experiment"
            )
            verified = verify_run_output(run_dir, protocol=spec["protocol"])
            upsert_result(report, attention, {**verified, "run_id": run_id,
                                              "state": "verified", "completed_at": iso_now(),
                                              "summary_run_id": summary.get("run_id") if isinstance(summary, dict) else None})
            json_write_atomic(report_path, report)
            print(f"[VERIFIED] {attention} | {run_id}", flush=True)

        # Re-check all attention variants before marking the original section as PASSED.
        all_verified = True
        for attention in ATTENTIONS:
            config = build_spec(spec, attention, args.seed)
            ok, result = inspect_run(output_root / config["run_id"], spec["protocol"])
            if ok:
                upsert_result(report, attention, {**result, "run_id": config["run_id"], "state": "verified"})
            else:
                all_verified = False
                # Keep history and prior successful entries; do not mark missing runs verified.
                upsert_result(report, attention, {
                    "run_id": config["run_id"], "attention": attention,
                    "state": "incomplete", "verification_error": result,
                })
        report["state"] = "passed" if all_verified else "incomplete"
        report["retry_history"][-1]["state"] = "passed" if all_verified else "incomplete"
        report["retry_history"][-1]["completed_at"] = iso_now()
        if all_verified:
            report["completed_at"] = iso_now()
            report.pop("error", None)
        json_write_atomic(report_path, report)
        print(f"SECTION {args.section} RECOVERY {'PASSED' if all_verified else 'INCOMPLETE'}")
        print(f"REPORT: {report_path}")
        return 0 if all_verified else 2
    except BaseException as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        if report is not None:
            report["state"] = "failed"
            report["last_retry_error"] = {"type": type(exc).__name__, "message": str(exc), "at": iso_now()}
            report["retry_history"][-1]["state"] = "failed"
            report["retry_history"][-1]["completed_at"] = iso_now()
            json_write_atomic(report_path, report)
        return 1
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=12)
        try:
            (session / f"retry_nvidia_smi_final_{args.attempt}.txt").write_text(
                command_snapshot(["nvidia-smi"]) + "\n", encoding="utf-8"
            )
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        retry_log.close()
        section_log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry only incomplete PASS attention runs inside their original production section")
    parser.add_argument("--machine", choices=("pc2", "server"), required=True)
    parser.add_argument("--section", type=int, required=True)
    parser.add_argument("--attentions", nargs="+", choices=ATTENTIONS, required=True)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--session-dir", default=None, help="Existing section directory under outputs/activity_logs; default latest matching directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.attentions) != len(set(args.attentions)):
        parser.error("Duplicate attention name detected")
    args.device = args.device if args.device is not None else (0 if args.machine == "pc2" else 1)
    args.num_workers = args.num_workers if args.num_workers is not None else (4 if args.machine == "pc2" else 8)
    sections = section_table(args.machine)
    if args.section not in sections:
        parser.error(f"Section {args.section} not found. Update scripts/production_section.py first.")
    # Refuse to use a non-PASS or wrong-section configuration even if the ID happens to exist.
    expected = {("pc2", 5): ("T5", "r18"), ("server", 13): ("T5", "r34")}
    if (args.machine, args.section) not in expected:
        parser.error("For this recovery, only pc2 Section 5 and server Section 13 are permitted")
    protocol, backbone = expected[(args.machine, args.section)]
    spec = sections[args.section]
    if (spec.get("method"), spec.get("dataset"), spec.get("protocol"), spec.get("backbone")) != ("pass", "aid", protocol, backbone):
        parser.error("Section mapping differs from expected PASS AID T5 configuration; stopping for safety")
    os.chdir(ROOT)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
