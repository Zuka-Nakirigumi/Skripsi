from pathlib import Path
import argparse
import sys

import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    force=True,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from thesis.experiments.config import (
    build_run_config,
    make_profile_overrides,
)

from thesis.experiments.runner import (
    run_experiment,
)


def main():

    parser = (
        argparse
        .ArgumentParser()
    )


    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "aid",
            "nwpu",
        ],
    )


    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "fetril",
            "fecam",
            "ssre",
            "pass",
            "efc",
        ],
    )


    parser.add_argument(
        "--protocol",
        required=True,
        choices=[
            "T5",
            "T10",
        ],
    )


    parser.add_argument(
        "--backbone",
        required=True,
        choices=[
            "r18",
            "r34",
        ],
    )


    parser.add_argument(
        "--attention",
        required=True,
        choices=[
            "none",
            "eca",
            "cbam",
            "procbam",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
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
        "--profile-epochs",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
    )


    args = (
        parser.parse_args()
    )


    overrides = {}

    if args.profile_epochs is not None:

        overrides.update(
            make_profile_overrides(
                args.method,
                args.profile_epochs,
            )
        )


    if args.batch_size is not None:

        overrides[
            "batch_size"
        ] = args.batch_size


    if not overrides:

        overrides = None


    resolved = (
        build_run_config(
            dataset=(
                args.dataset
            ),

            method=(
                args.method
            ),

            protocol=(
                args.protocol
            ),

            backbone=(
                args.backbone
            ),

            attention=(
                args.attention
            ),

            training_seed=(
                args.seed
            ),

            method_overrides=(
                overrides
            ),
        )
    )


    if (
        args.profile_epochs
        is None
    ):

        output_root = (
            "outputs/experiments"
        )

        run_mode = (
            "experiment"
        )

    else:

        output_root = (
            f"outputs/profiling/"
            f"{args.profile_epochs}ep"
        )

        run_mode = (
            "profiling"
        )


    run_experiment(
        resolved,
        output_root=(
            output_root
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
        run_mode=(
            run_mode
        ),
    )


if __name__ == "__main__":
    main()