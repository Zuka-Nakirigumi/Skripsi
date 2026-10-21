from __future__ import annotations

import copy
import json

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


VALID_DATASETS = {
    "aid",
    "nwpu",
}

VALID_METHODS = {
    "fetril",
    "fecam",
    "ssre",
    "pass",
    "efc",
}

VALID_BACKBONES = {
    "r18",
    "r34",
}

VALID_ATTENTIONS = {
    "none",
    "eca",
    "cbam",
    "procbam",
}

VALID_PROTOCOLS = {
    "T5",
    "T10",
}


def _read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def _resolve_project_path(
    value,
):

    if value is None:
        return None

    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def load_dataset_config(
    dataset_name: str,
) -> dict:

    dataset_name = (
        dataset_name
        .strip()
        .lower()
    )

    if dataset_name not in VALID_DATASETS:
        raise ValueError(
            "Unknown dataset: "
            f"{dataset_name}"
        )

    path = (
        PROJECT_ROOT
        / "configs"
        / "datasets"
        / f"{dataset_name}.json"
    )

    config = _read_json(
        path
    )

    config["_config_path"] = str(
        path
    )

    return config


def load_method_config(
    method_name: str,
) -> dict:

    method_name = (
        method_name
        .strip()
        .lower()
    )

    if method_name not in VALID_METHODS:
        raise ValueError(
            "Unknown method: "
            f"{method_name}"
        )

    path = (
        PROJECT_ROOT
        / "configs"
        / "methods"
        / f"{method_name}.json"
    )

    config = _read_json(
        path
    )

    config["_config_path"] = str(
        path
    )

    return config


def apply_overrides(
    config: dict,
    overrides: dict | None,
) -> dict:

    result = copy.deepcopy(
        config
    )

    if overrides is None:
        return result

    for key, value in overrides.items():

        if key not in result:
            raise KeyError(
                "Override attempted for "
                "unknown config key: "
                f"{key}"
            )

        result[key] = value

    return result

def validate_dataset_config(
    config: dict,
    protocol: str,
):

    protocol = protocol.upper()

    if protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"Invalid protocol: {protocol}"
        )

    required_keys = [
        "dataset",
        "num_classes",
        "input_size",
        "dataset_root",
        "train_manifest",
        "test_manifest",
        "task_configs",
        "normalization",
    ]

    for key in required_keys:

        if key not in config:
            raise KeyError(
                "Dataset config missing "
                f"required key: {key}"
            )


    dataset_name = (
        config["dataset"]
        .lower()
    )

    if dataset_name not in VALID_DATASETS:
        raise ValueError(
            "Unsupported dataset: "
            f"{dataset_name}"
        )


    if int(
        config["num_classes"]
    ) <= 0:

        raise ValueError(
            "num_classes must be > 0"
        )


    if int(
        config["input_size"]
    ) <= 0:

        raise ValueError(
            "input_size must be > 0"
        )


    normalization = (
        config[
            "normalization"
        ]
    )


    mean = normalization.get(
        "mean"
    )

    std = normalization.get(
        "std"
    )


    if (
        mean is None
        or std is None
    ):
        raise ValueError(
            "Normalization mean/std "
            "must be defined."
        )


    if (
        len(mean) != 3
        or len(std) != 3
    ):
        raise ValueError(
            "Normalization must contain "
            "3 RGB mean/std values."
        )


    if any(
        value <= 0
        for value in std
    ):
        raise ValueError(
            "All normalization std "
            "values must be > 0."
        )


    dataset_root = (
        _resolve_project_path(
            config[
                "dataset_root"
            ]
        )
    )

    train_manifest = (
        _resolve_project_path(
            config[
                "train_manifest"
            ]
        )
    )

    test_manifest = (
        _resolve_project_path(
            config[
                "test_manifest"
            ]
        )
    )


    task_path = (
        _resolve_project_path(
            config[
                "task_configs"
            ][
                protocol
            ]
        )
    )


    paths = {
        "dataset_root":
            dataset_root,

        "train_manifest":
            train_manifest,

        "test_manifest":
            test_manifest,

        "task_config":
            task_path,
    }


    for name, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: "
                f"{path}"
            )


    return paths

def validate_method_config(
    method_name: str,
    config: dict,
):

    method_name = (
        method_name
        .lower()
    )


    if method_name not in VALID_METHODS:
        raise ValueError(
            f"Invalid method: {method_name}"
        )


    if "batch_size" not in config:
        raise KeyError(
            "Method config requires "
            "'batch_size'."
        )


    if int(
        config["batch_size"]
    ) <= 0:

        raise ValueError(
            "batch_size must be > 0."
        )


    if method_name == "fetril":

        required = [
            "init_epochs",
            "init_lr",
            "init_weight_decay",
            "classifier",
            "svm_c",
            "svm_tol",
        ]


        if (
            config[
                "classifier"
            ]
            != "svm"
        ):
            raise ValueError(
                "Canonical thesis FeTrIL "
                "must use SVM."
            )


    elif method_name == "fecam":

        required = [
            "init_epochs",
            "init_lr",
            "init_weight_decay",
            "beta",
            "alpha1",
            "alpha2",
            "full_cov",
            "shrink",
            "norm_cov",
        ]


    elif method_name in {
        "ssre",
        "pass",
    }:

        required = [
            "epochs",
            "lr",
            "weight_decay",
            "lambda_fkd",
            "lambda_proto",
            "temp",
        ]


    elif method_name == "efc":

        required = [
            "epochs_first_task",
            "epochs_next_task",
            "efc_lamb",
            "efc_damping",
            "efc_protoupdate",
            "efc_protobatchsize",
        ]


    else:

        required = []


    for key in required:

        if key not in config:
            raise KeyError(
                f"{method_name} config "
                f"missing key: {key}"
            )


    epoch_keys = [
        key
        for key in [
            "init_epochs",
            "epochs",
            "epochs_first_task",
            "epochs_next_task",
        ]
        if key in config
    ]


    for key in epoch_keys:

        if int(
            config[key]
        ) <= 0:

            raise ValueError(
                f"{key} must be > 0."
            )


    return True

def build_run_config(
    dataset: str,
    method: str,
    protocol: str,
    backbone: str,
    attention: str,
    training_seed: int = 1993,
    method_overrides: dict | None = None,
):

    dataset = dataset.lower()

    method = method.lower()

    protocol = protocol.upper()

    backbone = backbone.lower()

    attention = attention.lower()


    if backbone not in VALID_BACKBONES:
        raise ValueError(
            f"Invalid backbone: {backbone}"
        )


    if attention not in VALID_ATTENTIONS:
        raise ValueError(
            f"Invalid attention: {attention}"
        )


    dataset_config = (
        load_dataset_config(
            dataset
        )
    )

    method_config = (
        load_method_config(
            method
        )
    )


    method_config = (
        apply_overrides(
            method_config,
            method_overrides,
        )
    )


    paths = (
        validate_dataset_config(
            dataset_config,
            protocol,
        )
    )


    validate_method_config(
        method,
        method_config,
    )


    run_id = (
        f"{dataset}"
        f"_{method}"
        f"_{protocol.lower()}"
        f"_{backbone}"
        f"_{attention}"
        f"_seed{training_seed}"
    )


    resolved = {
        "run_id":
            run_id,

        "dataset":
            dataset,

        "method":
            method,

        "protocol":
            protocol,

        "backbone":
            backbone,

        "attention":
            attention,

        "training_seed":
            int(
                training_seed
            ),

        "dataset_config":
            dataset_config,

        "method_config":
            method_config,

        "resolved_paths": {
            key: str(value)
            for key, value
            in paths.items()
        },
    }


    return resolved

def make_profile_overrides(
    method_name: str,
    profile_epochs: int = 5,
):

    if profile_epochs <= 0:
        raise ValueError(
            "profile_epochs must be > 0"
        )


    method_name = (
        method_name
        .lower()
    )


    if method_name in {
        "fetril",
        "fecam",
    }:

        return {
            "init_epochs":
                profile_epochs,
        }


    if method_name in {
        "ssre",
        "pass",
    }:

        return {
            "epochs":
                profile_epochs,
        }


    if method_name == "efc":

        return {
            "epochs_first_task":
                profile_epochs,

            "epochs_next_task":
                profile_epochs,
        }


    raise ValueError(
        "Unknown method for profiling: "
        f"{method_name}"
    )