#!/usr/bin/env python3
"""Verify imports for PyCIL production feature adapter (no checkpoint loading)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main():
    script = Path(__file__).resolve()
    repo = script.parent.parent
    adapter_file = script.parent / "feature_adapter_pycil_production.py"
    report = {
        "project_root": str(repo),
        "adapter_path": str(adapter_file),
        "thesis_loader_file_exists": (repo / "thesis" / "data" / "loaders.py").is_file(),
        "data_manager_file_exists": (repo / "utils" / "data_manager.py").is_file(),
        "status": "failed",
    }
    try:
        if not adapter_file.is_file():
            raise FileNotFoundError(f"Missing adapter: {adapter_file}")
        spec = importlib.util.spec_from_file_location("pycil_feature_adapter_check", adapter_file)
        if spec is None or spec.loader is None:
            raise ImportError("Could not import feature adapter")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        report["adapter_version"] = getattr(mod, "FEATURE_ADAPTER_VERSION", "legacy/missing")
        resolved = mod._bootstrap_pycil_imports()
        report["resolved_project_root"] = str(resolved)
        from thesis.data.loaders import build_eval_loader
        from utils.data_manager import DataManager
        report["eval_loader_importable"] = callable(build_eval_loader)
        report["data_manager_importable"] = callable(DataManager)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
