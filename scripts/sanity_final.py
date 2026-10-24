from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from thesis.experiments.config import (
    build_run_config,
    make_profile_overrides,
)

from thesis.experiments.runner import (
    run_experiment,
)


ACTIVE_METHODS = (
    "fetril",
    "fecam",
    "ssre",
    "efc",
)

DEFAULT_BATCH = {
    "fetril": 20,
    "fecam": 20,
    "ssre": 20,
    "efc": 4,
}

LEARNER_FILES = {
    "fetril":
        PROJECT_ROOT / "models" / "fetril.py",

    "fecam":
        PROJECT_ROOT / "models" / "fecam.py",

    "ssre":
        PROJECT_ROOT / "models" / "ssre.py",

    "efc":
        PROJECT_ROOT / "models" / "efc.py",
}


def parse_batch_overrides(
    values,
):
    result = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                "--batch must use "
                "METHOD=VALUE, e.g. "
                "--batch fecam=20"
            )

        method, batch = (
            value.split(
                "=",
                1,
            )
        )

        method = (
            method
            .strip()
            .lower()
        )

        if method not in ACTIVE_METHODS:
            raise ValueError(
                "Unsupported method in "
                f"--batch: {method}"
            )

        result[method] = int(
            batch
        )

    return result


def preflight_canonical_cache(
    methods,
):
    required_tokens = (
        "_last_canonical_y_pred",
        "_last_canonical_y_true",
        "_canonical_eval_name",
    )

    missing = []

    for method in methods:
        path = (
            LEARNER_FILES[
                method
            ]
        )

        if not path.exists():
            missing.append(
                f"{method}: file missing: "
                f"{path}"
            )
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        missing_tokens = [
            token
            for token in required_tokens
            if token not in text
        ]

        if missing_tokens:
            missing.append(
                f"{method}: missing "
                + ", ".join(
                    missing_tokens
                )
            )

    if missing:
        raise RuntimeError(
            "Canonical prediction cache "
            "preflight FAILED:\n- "
            + "\n- ".join(
                missing
            )
        )


def primary_top1(
    primary,
):
    if not isinstance(
        primary,
        dict,
    ):
        return None

    if primary.get(
        "top1"
    ) is not None:
        return float(
            primary[
                "top1"
            ]
        )

    metrics = primary.get(
        "metrics"
    )

    if (
        isinstance(
            metrics,
            dict,
        )
        and metrics.get(
            "top1"
        ) is not None
    ):
        return float(
            metrics[
                "top1"
            ]
        )

    return None


def require_file(
    path,
):
    path = Path(
        path
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing artifact: {path}"
        )

    return path


def verify_method_output(
    run_dir,
    expected_tasks,
):
    run_dir = Path(
        run_dir
    )

    status_path = require_file(
        run_dir
        / "status.json"
    )

    status = json.loads(
        status_path.read_text(
            encoding="utf-8"
        )
    )

    if status.get(
        "state"
    ) != "completed":
        raise RuntimeError(
            "Run did not complete: "
            f"{run_dir}"
        )

    effective_args = json.loads(
        require_file(
            run_dir
            / "effective_args.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    if not effective_args.get(
        "save_checkpoints",
        False,
    ):
        raise RuntimeError(
            "Sanity run did not have "
            "save_checkpoints=True."
        )

    require_file(
        run_dir
        / "provenance.json"
    )

    task_metrics = json.loads(
        require_file(
            run_dir
            / "task_metrics.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    if len(
        task_metrics
    ) != expected_tasks:
        raise RuntimeError(
            "Unexpected task count in "
            "task_metrics.json: "
            f"{len(task_metrics)} "
            f"!= {expected_tasks}"
        )

    continual = json.loads(
        require_file(
            run_dir
            / "continual_metrics.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    required_metrics = (
        "aia",
        "last_task_accuracy",
        "forgetting_rate",
        "bwt",
        "macro_f1_final",
        "joint_top1_history",
        "task_accuracy_matrix",
        "class_accuracy_matrix",
        "fwt_status",
    )

    for key in required_metrics:
        if key not in continual:
            raise RuntimeError(
                "Missing continual metric: "
                f"{key}"
            )

    if continual.get(
        "fwt"
    ) is not None:
        raise RuntimeError(
            "FWT must remain null under "
            "the current dynamic-classifier "
            "protocol."
        )

    require_file(
        run_dir
        / "summary.json"
    )

    require_file(
        run_dir
        / "final_confusion_matrix.npy"
    )

    checkpoint_dir = require_file(
        run_dir
        / "checkpoints"
    )

    for task_id in range(
        expected_tasks
    ):
        prediction_path = require_file(
            run_dir
            / "predictions"
            / f"task_{task_id:02d}.npz"
        )

        prediction_data = np.load(
            prediction_path
        )

        if (
            "y_true"
            not in prediction_data
            or "y_pred_topk"
            not in prediction_data
        ):
            raise RuntimeError(
                "Prediction archive missing "
                f"arrays: {prediction_path}"
            )

        y_true = (
            prediction_data[
                "y_true"
            ]
        )

        y_pred = (
            prediction_data[
                "y_pred_topk"
            ]
        )

        if (
            y_pred.shape[0]
            != y_true.shape[0]
        ):
            raise RuntimeError(
                "Prediction length mismatch "
                f"for task {task_id}."
            )

        if y_pred.ndim == 2:
            top1 = y_pred[:, 0]
        else:
            top1 = y_pred

        recomputed_top1 = float(
            np.mean(
                top1
                == y_true
            )
            * 100.0
        )

        stored_canonical = float(
            task_metrics[
                task_id
            ][
                "canonical_top1"
            ]
        )

        if abs(
            recomputed_top1
            - stored_canonical
        ) > 1e-8:
            raise RuntimeError(
                "Stored canonical_top1 "
                "does not match saved "
                f"predictions on task {task_id}."
            )

        primary = primary_top1(
            task_metrics[
                task_id
            ][
                "primary"
            ]
        )

        if (
            primary is not None
            and abs(
                primary
                - stored_canonical
            ) > 0.051
        ):
            raise RuntimeError(
                "Primary/canonical mismatch "
                f"on task {task_id}: "
                f"primary={primary}, "
                f"canonical={stored_canonical}"
            )

        require_file(
            run_dir
            / "confusion_matrices"
            / f"task_{task_id:02d}.npy"
        )

        network_path = require_file(
            checkpoint_dir
            / f"model_{task_id}.pkl"
        )

        network_state = torch.load(
            network_path,
            map_location="cpu",
        )

        if (
            "model_state_dict"
            not in network_state
        ):
            raise RuntimeError(
                "Network checkpoint missing "
                f"model_state_dict: {network_path}"
            )

        method_state_path = require_file(
            checkpoint_dir
            / (
                f"method_state_"
                f"{task_id}.pkl"
            )
        )

        with method_state_path.open(
            "rb"
        ) as file:
            method_state = pickle.load(
                file
            )

        if int(
            method_state.get(
                "task_id",
                -1,
            )
        ) != task_id:
            raise RuntimeError(
                "Method-state task_id "
                f"mismatch: {method_state_path}"
            )

        manifest = json.loads(
            require_file(
                checkpoint_dir
                / (
                    f"checkpoint_"
                    f"{task_id}.json"
                )
            )
            .read_text(
                encoding="utf-8"
            )
        )

        if (
            "network_checkpoint"
            not in manifest
            or "method_state"
            not in manifest
        ):
            raise RuntimeError(
                "Checkpoint manifest "
                f"incomplete for task {task_id}."
            )

    return {
        "run_dir":
            str(
                run_dir
            ),

        "tasks":
            expected_tasks,

        "aia":
            continual[
                "aia"
            ],

        "last_task_accuracy":
            continual[
                "last_task_accuracy"
            ],

        "macro_f1_final":
            continual[
                "macro_f1_final"
            ],

        "forgetting_rate":
            continual[
                "forgetting_rate"
            ],

        "bwt":
            continual[
                "bwt"
            ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(
            ACTIVE_METHODS
        ),
        choices=list(
            ACTIVE_METHODS
        ),
    )

    parser.add_argument(
        "--dataset",
        default="aid",
        choices=[
            "aid",
            "nwpu",
        ],
    )

    parser.add_argument(
        "--protocol",
        default="T5",
        choices=[
            "T5",
            "T10",
        ],
    )

    parser.add_argument(
        "--backbone",
        default="r18",
        choices=[
            "r18",
            "r34",
        ],
    )

    parser.add_argument(
        "--attention",
        default="none",
        choices=[
            "none",
            "eca",
            "cbam",
            "procbam",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
    )

    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        help=(
            "Repeatable METHOD=VALUE "
            "batch override."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=None,
    )

    args = parser.parse_args()

    methods = [
        method.lower()
        for method in args.methods
    ]

    preflight_canonical_cache(
        methods
    )

    batch_override = (
        parse_batch_overrides(
            args.batch
        )
    )

    if args.output_root is None:
        stamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        output_root = (
            PROJECT_ROOT
            / "outputs"
            / "sanity_final"
            / stamp
        )

    else:
        output_root = Path(
            args.output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "state":
            "running",

        "started_at":
            datetime.now()
            .isoformat(),

        "methods":
            methods,

        "dataset":
            args.dataset,

        "protocol":
            args.protocol,

        "backbone":
            args.backbone,

        "attention":
            args.attention,

        "epochs":
            args.epochs,

        "max_tasks":
            args.max_tasks,

        "device":
            args.device,

        "results":
            [],
    }

    report_path = (
        output_root
        / "sanity_report.json"
    )

    try:
        for method in methods:
            batch = int(
                batch_override.get(
                    method,
                    DEFAULT_BATCH[
                        method
                    ],
                )
            )

            overrides = (
                make_profile_overrides(
                    method,
                    args.epochs,
                )
            )

            if overrides is None:
                overrides = {}

            overrides = dict(
                overrides
            )

            overrides[
                "batch_size"
            ] = batch

            resolved = (
                build_run_config(
                    dataset=args.dataset,
                    method=method,
                    protocol=args.protocol,
                    backbone=args.backbone,
                    attention=args.attention,
                    training_seed=args.seed,
                    method_overrides=(
                        overrides
                    ),
                )
            )

            print()
            print(
                "=" * 72
            )
            print(
                "FINAL SANITY"
            )
            print(
                "METHOD    :",
                method,
            )
            print(
                "BACKBONE  :",
                args.backbone,
            )
            print(
                "ATTENTION :",
                args.attention,
            )
            print(
                "BATCH     :",
                batch,
            )
            print(
                "EPOCHS    :",
                args.epochs,
            )
            print(
                "TASKS     :",
                args.max_tasks,
            )
            print(
                "=" * 72
            )

            summary = (
                run_experiment(
                    resolved,
                    output_root=(
                        str(
                            output_root
                        )
                    ),
                    device_index=(
                        args.device
                    ),
                    num_workers=(
                        args.num_workers
                    ),
                    max_tasks=(
                        args.max_tasks
                    ),
                    run_mode="sanity",
                )
            )

            run_dir = (
                output_root
                / resolved[
                    "run_id"
                ]
            )

            verified = (
                verify_method_output(
                    run_dir,
                    expected_tasks=(
                        args.max_tasks
                    ),
                )
            )

            verified[
                "method"
            ] = method

            verified[
                "batch_size"
            ] = batch

            report[
                "results"
            ].append(
                verified
            )

            report[
                "last_summary_run_id"
            ] = summary.get(
                "run_id"
            )

            report_path.write_text(
                json.dumps(
                    report,
                    indent=2,
                ),
                encoding="utf-8",
            )

        report[
            "state"
        ] = "passed"

        report[
            "completed_at"
        ] = (
            datetime.now()
            .isoformat()
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "=" * 72
        )
        print(
            "FINAL SANITY: PASSED"
        )
        print(
            "REPORT:",
            report_path,
        )
        print(
            "=" * 72
        )

    except Exception as exc:
        report[
            "state"
        ] = "failed"

        report[
            "completed_at"
        ] = (
            datetime.now()
            .isoformat()
        )

        report[
            "error"
        ] = {
            "type":
                type(
                    exc
                ).__name__,

            "message":
                str(
                    exc
                ),
        }

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise


if __name__ == "__main__":
    main()
