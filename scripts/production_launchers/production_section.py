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
)

from thesis.experiments.runner import (
    run_experiment,
)


ATTENTIONS = (
    "none",
    "eca",
    "cbam",
    "procbam",
)


PC2_SECTIONS = {
    1: {
        "label": "FeCAM AID T5 R18",
        "method": "fecam",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r18",
        "batch_size": 20,
    },
    2: {
        "label": "FeCAM AID T10 R18",
        "method": "fecam",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r18",
        "batch_size": 20,
    },
    3: {
        "label": "SSRE AID T5 R18",
        "method": "ssre",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r18",
        "batch_size": 20,
    },
    4: {
        "label": "SSRE AID T10 R18",
        "method": "ssre",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r18",
        "batch_size": 20,
    },
    5: {
        "label": "PASS AID T5 R18",
        "method": "pass",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r18",
        "batch_size": 4,
    },
    6: {
        "label": "PASS AID T10 R18",
        "method": "pass",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r18",
        "batch_size": 4,
    },
}


SERVER_SECTIONS = {
    1: {
        "label": "FeTrIL AID T5 R18",
        "method": "fetril",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r18",
        "batch_size": 32,
    },
    2: {
        "label": "FeTrIL AID T10 R18",
        "method": "fetril",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r18",
        "batch_size": 32,
    },
    3: {
        "label": "FeTrIL AID T5 R34",
        "method": "fetril",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r34",
        "batch_size": 32,
    },
    4: {
        "label": "FeTrIL AID T10 R34",
        "method": "fetril",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r34",
        "batch_size": 32,
    },
    5: {
        "label": "EFC AID T5 R18",
        "method": "efc",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r18",
        "batch_size": 12,
    },
    6: {
        "label": "EFC AID T10 R18",
        "method": "efc",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r18",
        "batch_size": 12,
    },
    7: {
        "label": "EFC AID T5 R34",
        "method": "efc",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r34",
        "batch_size": 12,
    },
    8: {
        "label": "EFC AID T10 R34",
        "method": "efc",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r34",
        "batch_size": 12,
    },
    9: {
        "label": "FeCAM AID T5 R34",
        "method": "fecam",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r34",
        "batch_size": 64,
    },
    10: {
        "label": "FeCAM AID T10 R34",
        "method": "fecam",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r34",
        "batch_size": 64,
    },
    11: {
        "label": "SSRE AID T5 R34",
        "method": "ssre",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r34",
        "batch_size": 32,
    },
    12: {
        "label": "SSRE AID T10 R34",
        "method": "ssre",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r34",
        "batch_size": 32,
    },
    13: {
        "label": "PASS AID T5  R34",
        "method": "pass",
        "dataset": "aid",
        "protocol": "T5",
        "backbone": "r34",
        "batch_size": 12,
    },
    14: {
        "label": "PASS AID T10 R34",
        "method": "pass",
        "dataset": "aid",
        "protocol": "T10",
        "backbone": "r34",
        "batch_size": 12,
    },
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
    "pass": 
        PROJECT_ROOT / "models" / "pa2s.py",
}


def write_json(
    path,
    data,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def require_file(
    path,
):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"Missing artifact: {path}"
        )

    return path


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
            primary["top1"]
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
            metrics["top1"]
        )

    return None


def expected_tasks(
    protocol,
):
    if protocol == "T5":
        return 5

    if protocol == "T10":
        return 10

    raise ValueError(
        f"Unsupported protocol: {protocol}"
    )


def preflight_canonical_cache(
    method,
):
    path = LEARNER_FILES[
        method
    ]

    if not path.exists():
        raise RuntimeError(
            f"Learner file missing: {path}"
        )

    content = path.read_text(
        encoding="utf-8"
    )

    required_tokens = (
        "_last_canonical_y_pred",
        "_last_canonical_y_true",
        "_canonical_eval_name",
    )

    missing = [
        token
        for token in required_tokens
        if token not in content
    ]

    if missing:
        raise RuntimeError(
            f"{method} canonical cache "
            "preflight failed; missing: "
            + ", ".join(missing)
        )


def verify_run_output(
    run_dir,
    protocol,
):
    run_dir = Path(
        run_dir
    )

    n_tasks = expected_tasks(
        protocol
    )

    status = json.loads(
        require_file(
            run_dir
            / "status.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    if status.get(
        "state"
    ) != "completed":
        raise RuntimeError(
            f"Run is not completed: {run_dir}"
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
            "Production run has "
            "save_checkpoints != True."
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
    ) != n_tasks:
        raise RuntimeError(
            "Unexpected task count in "
            f"{run_dir}: "
            f"{len(task_metrics)} != {n_tasks}"
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

    require_file(
        run_dir
        / "summary.json"
    )

    final_confusion = np.load(
        require_file(
            run_dir
            / "final_confusion_matrix.npy"
        )
    )

    if final_confusion.ndim != 2:
        raise RuntimeError(
            "Final confusion matrix "
            "must be 2-dimensional."
        )

    checkpoint_dir = require_file(
        run_dir
        / "model_checkpoints"
    )

    for task_id in range(
        n_tasks
    ):
        metric = task_metrics[
            task_id
        ]

        prediction_path = require_file(
            run_dir
            / "predictions"
            / f"task_{task_id:02d}.npz"
        )

        with np.load(
            prediction_path
        ) as prediction_data:
            if (
                "y_true"
                not in prediction_data
                or "y_pred_topk"
                not in prediction_data
            ):
                raise RuntimeError(
                    "Prediction archive missing "
                    f"required arrays: {prediction_path}"
                )

            y_true = np.asarray(
                prediction_data[
                    "y_true"
                ]
            )

            y_pred = np.asarray(
                prediction_data[
                    "y_pred_topk"
                ]
            )

        if (
            y_pred.shape[0]
            != y_true.shape[0]
        ):
            raise RuntimeError(
                "Prediction/target length "
                f"mismatch on task {task_id}."
            )

        top1 = (
            y_pred[:, 0]
            if y_pred.ndim == 2
            else y_pred
        )

        recomputed_top1 = float(
            np.mean(
                top1
                == y_true
            )
            * 100.0
        )

        stored_canonical = float(
            metric[
                "canonical_top1"
            ]
        )

        if abs(
            recomputed_top1
            - stored_canonical
        ) > 1e-8:
            raise RuntimeError(
                "Saved predictions do not "
                "match canonical_top1 on "
                f"task {task_id}."
            )

        primary = primary_top1(
            metric.get(
                "primary"
            )
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
                f"{primary} vs "
                f"{stored_canonical}"
            )

        confusion_path = require_file(
            run_dir
            / "confusion_matrices"
            / f"task_{task_id:02d}.npy"
        )

        confusion = np.load(
            confusion_path
        )

        seen_classes = int(
            metric[
                "seen_classes"
            ]
        )

        if confusion.shape != (
            seen_classes,
            seen_classes,
        ):
            raise RuntimeError(
                "Unexpected confusion "
                f"shape on task {task_id}: "
                f"{confusion.shape}"
            )

        if int(
            confusion.sum()
        ) != int(
            y_true.shape[0]
        ):
            raise RuntimeError(
                "Confusion matrix sample "
                "count mismatch on task "
                f"{task_id}."
            )

        network_path = require_file(
            checkpoint_dir
            / f"model_{task_id}.pkl"
        )

        network_state = torch.load(
            network_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            not isinstance(
                network_state,
                dict,
            )
            or "model_state_dict"
            not in network_state
        ):
            raise RuntimeError(
                "Invalid network checkpoint: "
                f"{network_path}"
            )

        method_state_path = require_file(
            checkpoint_dir
            / f"method_state_{task_id}.pkl"
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
                "Method-state task id "
                f"mismatch: {method_state_path}"
            )

        manifest = json.loads(
            require_file(
                checkpoint_dir
                / f"checkpoint_{task_id}.json"
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
                "Incomplete checkpoint "
                f"manifest on task {task_id}."
            )

    required_continual = (
        "aia",
        "last_task_accuracy",
        "forgetting_rate",
        "bwt",
        "macro_f1_final",
        "task_accuracy_matrix",
        "class_accuracy_matrix",
    )

    missing_metrics = [
        key
        for key in required_continual
        if key not in continual
    ]

    if missing_metrics:
        raise RuntimeError(
            "continual_metrics.json "
            "missing: "
            + ", ".join(
                missing_metrics
            )
        )

    return {
        "run_dir":
            str(run_dir),

        "tasks":
            n_tasks,

        "aia":
            continual.get(
                "aia"
            ),

        "last_task_accuracy":
            continual.get(
                "last_task_accuracy"
            ),

        "macro_f1_final":
            continual.get(
                "macro_f1_final"
            ),

        "state":
            "verified",
    }


def section_table(
    machine,
):
    if machine == "pc2":
        return PC2_SECTIONS

    if machine == "server":
        return SERVER_SECTIONS

    raise ValueError(
        f"Unsupported machine: {machine}"
    )


def print_sections(
    machine,
):
    table = section_table(
        machine
    )

    print(
        f"{machine.upper()} PRODUCTION SECTIONS"
    )

    for section_id, spec in (
        table.items()
    ):
        print(
            f"{section_id:>2}: "
            f"{spec['label']} | "
            f"batch={spec['batch_size']} | "
            "attention="
            + ",".join(
                ATTENTIONS
            )
        )


def main():
    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--machine",
        required=True,
        choices=[
            "pc2",
            "server",
        ],
    )

    parser.add_argument(
        "--section",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--list-sections",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
    )

    parser.add_argument(
        "--output-root",
        default="outputs/experiments",
    )

    parser.add_argument(
        "--report-path",
        default=None,
    )

    args = (
        parser.parse_args()
    )

    if args.list_sections:
        print_sections(
            args.machine
        )
        return

    table = section_table(
        args.machine
    )

    if (
        args.section is None
        or args.section not in table
    ):
        raise ValueError(
            "Invalid section. Use "
            "--list-sections to inspect "
            "available sections."
        )

    spec = dict(
        table[
            args.section
        ]
    )

    if args.machine == "pc2":
        device = (
            0
            if args.device is None
            else args.device
        )
        num_workers = (
            4
            if args.num_workers is None
            else args.num_workers
        )
    else:
        device = (
            1
            if args.device is None
            else args.device
        )
        num_workers = (
            8
            if args.num_workers is None
            else args.num_workers
        )

    preflight_canonical_cache(
        spec[
            "method"
        ]
    )

    output_root = Path(
        args.output_root
    )

    report_path = (
        Path(args.report_path)
        if args.report_path
        else (
            output_root
            / (
                f"section_{args.machine}_"
                f"{args.section:02d}.json"
            )
        )
    )

    report = {
        "state":
            "running",

        "machine":
            args.machine,

        "section":
            args.section,

        "label":
            spec[
                "label"
            ],

        "dataset":
            spec[
                "dataset"
            ],

        "method":
            spec[
                "method"
            ],

        "protocol":
            spec[
                "protocol"
            ],

        "backbone":
            spec[
                "backbone"
            ],

        "batch_size":
            spec[
                "batch_size"
            ],

        "device":
            device,

        "num_workers":
            num_workers,

        "seed":
            args.seed,

        "attentions":
            list(
                ATTENTIONS
            ),

        "started_at":
            datetime.now()
            .isoformat(),

        "completed_at":
            None,

        "results":
            [],
    }

    write_json(
        report_path,
        report,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "PRODUCTION SECTION"
    )
    print(
        "MACHINE   :",
        args.machine,
    )
    print(
        "SECTION   :",
        args.section,
    )
    print(
        "LABEL     :",
        spec[
            "label"
        ],
    )
    print(
        "BATCH     :",
        spec[
            "batch_size"
        ],
    )
    print(
        "DEVICE    :",
        device,
    )
    print(
        "WORKERS   :",
        num_workers,
    )
    print(
        "ATTENTION :",
        ", ".join(
            ATTENTIONS
        ),
    )
    print(
        "=" * 72
    )

    try:
        for attention in ATTENTIONS:
            overrides = {
                "batch_size":
                    int(
                        spec[
                            "batch_size"
                        ]
                    ),
            }

            resolved = (
                build_run_config(
                    dataset=(
                        spec[
                            "dataset"
                        ]
                    ),
                    method=(
                        spec[
                            "method"
                        ]
                    ),
                    protocol=(
                        spec[
                            "protocol"
                        ]
                    ),
                    backbone=(
                        spec[
                            "backbone"
                        ]
                    ),
                    attention=(
                        attention
                    ),
                    training_seed=(
                        args.seed
                    ),
                    method_overrides=(
                        overrides
                    ),
                )
            )

            run_id = resolved[
                "run_id"
            ]

            print()
            print(
                "#" * 72
            )
            print(
                "PRODUCTION RUN"
            )
            print(
                "RUN       :",
                run_id,
            )
            print(
                "ATTENTION :",
                attention,
            )
            print(
                "CHECKPOINT: ON"
            )
            print(
                "#" * 72
            )

            run_record = {
                "run_id":
                    run_id,

                "attention":
                    attention,

                "state":
                    "running",

                "started_at":
                    datetime.now()
                    .isoformat(),

                "completed_at":
                    None,
            }

            report[
                "results"
            ].append(
                run_record
            )

            write_json(
                report_path,
                report,
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
                        device
                    ),
                    num_workers=(
                        num_workers
                    ),
                    max_tasks=None,
                    run_mode="experiment",
                )
            )

            run_dir = (
                output_root
                / run_id
            )

            verified = (
                verify_run_output(
                    run_dir,
                    protocol=(
                        spec[
                            "protocol"
                        ]
                    ),
                )
            )

            run_record.update(
                verified
            )

            run_record[
                "summary_run_id"
            ] = summary.get(
                "run_id"
            )

            run_record[
                "state"
            ] = "verified"

            run_record[
                "completed_at"
            ] = (
                datetime.now()
                .isoformat()
            )

            write_json(
                report_path,
                report,
            )

            print(
                "VERIFIED  :",
                run_id,
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

        write_json(
            report_path,
            report,
        )

        print()
        print(
            "=" * 72
        )
        print(
            "PRODUCTION SECTION PASSED"
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
                type(exc).__name__,

            "message":
                str(exc),
        }

        write_json(
            report_path,
            report,
        )

        raise


if __name__ == "__main__":
    main()
