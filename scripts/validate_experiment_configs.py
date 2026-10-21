from pathlib import Path
import sys


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


DATASETS = [
    "aid",
    "nwpu",
]

METHODS = [
    "fetril",
    "fecam",
    "ssre",
    "pass",
    "efc",
]

PROTOCOLS = [
    "T5",
    "T10",
]

BACKBONES = [
    "r18",
    "r34",
]

ATTENTIONS = [
    "none",
    "eca",
    "cbam",
    "procbam",
]


def main():

    count = 0


    for dataset in DATASETS:

        for method in METHODS:

            for protocol in PROTOCOLS:

                for backbone in BACKBONES:

                    for attention in ATTENTIONS:

                        config = (
                            build_run_config(
                                dataset=dataset,
                                method=method,
                                protocol=protocol,
                                backbone=backbone,
                                attention=attention,
                                training_seed=1993,
                            )
                        )


                        assert (
                            config[
                                "run_id"
                            ]
                        )

                        count += 1


    print(
        "Canonical configurations:",
        count,
    )


    assert count == 160


    profile_count = 0


    for dataset in DATASETS:

        for method in METHODS:

            overrides = (
                make_profile_overrides(
                    method,
                    profile_epochs=5,
                )
            )


            config = (
                build_run_config(
                    dataset=dataset,
                    method=method,
                    protocol="T5",
                    backbone="r18",
                    attention="none",
                    training_seed=1993,
                    method_overrides=overrides,
                )
            )


            print(
                dataset,
                method,
                config[
                    "method_config"
                ],
            )


            profile_count += 1


    assert profile_count == 10


    print()
    print(
        "CONFIG VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()