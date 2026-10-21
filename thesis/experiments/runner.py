from __future__ import annotations

import json
import random
import time
import traceback

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager


def _json_default(
    value,
):

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        torch.device,
    ):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    raise TypeError(
        f"Cannot serialize: "
        f"{type(value)}"
    )


def _write_json(
    path,
    data,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            default=_json_default,
        )


def set_random_seed(
    seed,
):
    seed = int(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_convnet_type(
    method_name,
    backbone,
):

    method_name = (
        method_name
        .lower()
    )

    backbone = (
        backbone
        .lower()
    )


    if backbone not in {
        "r18",
        "r34",
    }:

        raise ValueError(
            "Unsupported backbone: "
            f"{backbone}"
        )


    # ==========================================
    # SSRE requires adapter-compatible backbone
    # ==========================================

    if method_name == "ssre":

        mapping = {
            "r18":
                "remote_ssre_resnet18",

            "r34":
                "remote_ssre_resnet34",
        }


    # ==========================================
    # FeTrIL / FeCAM / PASS / EFC
    # ==========================================

    else:

        mapping = {
            "r18":
                "remote_resnet18",

            "r34":
                "remote_resnet34",
        }


    return mapping[
        backbone
    ]

def build_pycil_args(
    resolved_config,
    device_index=0,
    num_workers=4,
):
    dataset_config = (
        resolved_config[
            "dataset_config"
        ]
    )

    method_config = (
        resolved_config[
            "method_config"
        ]
    )

    backbone = (
        resolved_config[
            "backbone"
        ]
        .lower()
    )

    attention_type = (
        resolved_config[
            "attention"
        ]
        .lower()
    )

    method_name = (
        resolved_config[
            "method"
        ]
        .lower()
    )


    convnet_type = (
        resolve_convnet_type(
            method_name=method_name,
            backbone=backbone,
        )
    )

    if torch.cuda.is_available():
        gpu_count = (
            torch.cuda.device_count()
        )

        if (
            device_index < 0
            or device_index >= gpu_count
        ):

            raise ValueError(
                f"Invalid CUDA device index "
                f"{device_index}. "
                f"Available CUDA devices: "
                f"{gpu_count}"
            )

        device = torch.device(
            f"cuda:{device_index}"
        )

    else:

        device = torch.device(
            "cpu"
        )


    args = {
        "prefix":
            "thesis",

        "dataset":
            resolved_config[
                "dataset"
            ],

        "model_name":
            method_name,

        "convnet_type":
            convnet_type,

        "attention_type":
            attention_type,

        "memory_size":
            0,

        "fixed_memory":
            False,

        "shuffle":
            False,

        "seed":
            int(
                resolved_config[
                    "training_seed"
                ]
            ),

        "device": [
            device
        ],

        "num_workers":
            int(
                num_workers
            ),

        "dataset_root":
            resolved_config[
                "resolved_paths"
            ][
                "dataset_root"
            ],

        "split_root":
            str(
                Path(
                    resolved_config[
                        "resolved_paths"
                    ][
                        "train_manifest"
                    ]
                )
                .parent
            ),

        "task_config_path":
            resolved_config[
                "resolved_paths"
            ][
                "task_config"
            ],

        "normalization_mean":
            dataset_config[
                "normalization"
            ][
                "mean"
            ],

        "normalization_std":
            dataset_config[
                "normalization"
            ][
                "std"
            ],
    }


    for key, value in (
        method_config.items()
    ):

        if key.startswith(
            "_"
        ):
            continue

        args[key] = value


    return args

def build_data_manager(
    args,
):

    data_manager = DataManager(
        args[
            "dataset"
        ],
        args[
            "shuffle"
        ],
        args[
            "seed"
        ],

        0,
        0,

        dataset_root=(
            args[
                "dataset_root"
            ]
        ),

        split_root=(
            args[
                "split_root"
            ]
        ),

        task_config_path=(
            args[
                "task_config_path"
            ]
        ),
    )


    args[
        "init_cls"
    ] = (
        data_manager
        .get_task_size(
            0
        )
    )


    if (
        data_manager.nb_tasks
        > 1
    ):

        args[
            "increment"
        ] = (
            data_manager
            .get_task_size(
                1
            )
        )

    else:

        args[
            "increment"
        ] = 0


    return data_manager

def get_primary_metric(
    method_name,
    model,
    eval_result,
):
    method_name = (
        method_name
        .lower()
    )


    cnn_accy = None
    secondary_accy = None


    if (
        isinstance(
            eval_result,
            tuple,
        )
        and len(
            eval_result
        ) == 2
    ):

        cnn_accy, secondary_accy = (
            eval_result
        )

    else:

        cnn_accy = (
            eval_result
        )


    if method_name == "fecam":

        return {
            "name":
                "fecam_mahalanobis",

            "metrics":
                secondary_accy,

            "cnn_diagnostic":
                cnn_accy,
        }


    if method_name == "fetril":

        svm_acc = None

        if hasattr(
            model,
            "_svm_accs",
        ):

            if len(
                model._svm_accs
            ) > 0:

                svm_acc = float(
                    model
                    ._svm_accs[
                        -1
                    ]
                )


        return {
            "name":
                "fetril_svm",

            "top1":
                svm_acc,

            "cnn_diagnostic":
                cnn_accy,
        }


    return {
        "name":
            "cnn",

        "metrics":
            cnn_accy,

        "secondary_diagnostic":
            secondary_accy,
    }

def run_experiment(
    resolved_config,
    output_root="outputs/experiments",
    device_index=0,
    num_workers=4,
    max_tasks=None,
    run_mode="experiment",
):

    run_id = (
        resolved_config[
            "run_id"
        ]
    )


    output_dir = (
        Path(
            output_root
        )
        / run_id
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    status_path = (
        output_dir
        / "status.json"
    )


    config_path = (
        output_dir
        / "config.json"
    )


    _write_json(
        config_path,
        resolved_config,
    )


    started_at = (
        datetime.now()
        .isoformat()
    )


    status = {
        "run_id":
            run_id,

        "mode":
            run_mode,

        "state":
            "running",

        "started_at":
            started_at,

        "completed_at":
            None,

        "error":
            None,
    }


    _write_json(
        status_path,
        status,
    )


    try:

        seed = (
            resolved_config[
                "training_seed"
            ]
        )

        set_random_seed(
            seed
        )


        args = (
            build_pycil_args(
                resolved_config,
                device_index=(
                    device_index
                ),
                num_workers=(
                    num_workers
                ),
            )
        )
        cuda_device = None
        if torch.cuda.is_available():
            cuda_device = (args["device"][0])
            torch.cuda.set_device(cuda_device)
        args["save_checkpoints"] = False


        data_manager = (
            build_data_manager(
                args
            )
        )

        print()     
        print("Resolved configuration")
        print("  dataset      :",args["dataset"],)
        print("  method       :",args["model_name"],)
        print("  convnet_type :",args["convnet_type"],)
        print("  attention    :",args["attention_type"],)
        print("  batch_size   :",args["batch_size"],)
        print("  device       :",args["device"],)

        model = (
            factory.get_model(
                args[
                    "model_name"
                ],
                args,
            )
        )


        task_count = (
            data_manager.nb_tasks
        )


        if max_tasks is not None:

            task_count = min(
                task_count,
                int(
                    max_tasks
                ),
            )


        run_start = (
            time.perf_counter()
        )


        task_results = []


        for task_id in range(
            task_count
        ):

            print()
            print(
                "=" * 70
            )

            print(
                f"RUN      : {run_id}"
            )

            print(
                f"TASK     : "
                f"{task_id + 1}"
                f"/{task_count}"
            )

            print(
                f"METHOD   : "
                f"{args['model_name']}"
            )

            print(
                f"BACKBONE : "
                f"{args['convnet_type']}"
            )

            print(
                "=" * 70
            )


            if cuda_device is not None:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(cuda_device)


            task_start = (
                time.perf_counter()
            )

            # ==========================================
            # Training
            # ==========================================

            train_start = (
                time.perf_counter()
            )

            model.incremental_train(
                data_manager
            )

            train_seconds = (
                time.perf_counter()
                - train_start
            )

            # ==========================================
            # Evaluation
            # ==========================================

            eval_start = (
                time.perf_counter()
            )

            eval_result = (
                model.eval_task()
            )

            eval_seconds = (
                time.perf_counter()
                - eval_start
            )


            primary = (
                get_primary_metric(
                    args[
                        "model_name"
                    ],
                    model,
                    eval_result,
                )
            )


            # ==========================================
            # After-task lifecycle
            # ==========================================

            after_task_start = (
                time.perf_counter()
            )

            model.after_task()

            after_task_seconds = (
                time.perf_counter()
                - after_task_start
            )


            task_seconds = (
                time.perf_counter()
                - task_start
            )


            if cuda_device is not None:
                peak_allocated = (
                    torch.cuda
                    .max_memory_allocated(
                        cuda_device
                    )
                    / (1024 ** 3)
                )
                peak_reserved = (
                    torch.cuda
                    .max_memory_reserved(
                        cuda_device
                    )
                    / (1024 ** 3)
                )

            else:

                peak_allocated = None
                peak_reserved = None


            primary = (
                get_primary_metric(
                    args[
                        "model_name"
                    ],
                    model,
                    eval_result,
                )
            )


            task_result = {
                "task_id":
                    task_id,

                "seen_classes":
                    int(
                        model._total_classes
                    ),

                "train_seconds":
                    train_seconds,

                "eval_seconds":
                    eval_seconds,

                "after_task_seconds":
                    after_task_seconds,

                "runtime_seconds":
                    task_seconds,

                "runtime_minutes":
                    task_seconds
                    / 60.0,

                "peak_vram_allocated_gb":
                    peak_allocated,

                "peak_vram_reserved_gb":
                    peak_reserved,

                "primary":
                    primary,
            }


            task_results.append(
                task_result
            )


            _write_json(
                output_dir
                / "task_metrics.json",
                task_results,
            )


        total_seconds = (
            time.perf_counter()
            - run_start
        )


        summary = {
            "run_id":
                run_id,

            "mode":
                run_mode,

            "tasks_completed":
                len(
                    task_results
                ),

            "total_runtime_seconds":
                total_seconds,

            "total_runtime_minutes":
                total_seconds
                / 60.0,

            "task_results":
                task_results,

            "device":
                (
                    torch.cuda
                    .get_device_name(
                        cuda_device
                    )
                    if cuda_device is not None
                    else "CPU"
                ),
        }


        _write_json(
            output_dir
            / "summary.json",
            summary,
        )


        status[
            "state"
        ] = "completed"

        status[
            "completed_at"
        ] = (
            datetime.now()
            .isoformat()
        )


        _write_json(
            status_path,
            status,
        )


        return summary


    except Exception as exc:

        status[
            "state"
        ] = "failed"

        status[
            "completed_at"
        ] = (
            datetime.now()
            .isoformat()
        )

        status[
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

            "traceback":
                traceback.format_exc(),
        }


        _write_json(
            status_path,
            status,
        )

        raise