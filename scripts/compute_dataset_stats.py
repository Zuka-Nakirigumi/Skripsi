from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm


def compute_stats(
    dataset_root,
    train_manifest,
):
    dataset_root = Path(
        dataset_root
    )

    train_manifest = Path(
        train_manifest
    )


    df = pd.read_csv(
        train_manifest
    )


    channel_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    channel_sq_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    pixel_count = 0


    for relative_path in tqdm(
        df["relative_path"]
    ):

        relative_path = (
            str(relative_path)
            .replace(
                "\\",
                "/",
            )
        )

        image_path = (
            dataset_root
            / relative_path
        )


        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            arr = (
                np.asarray(
                    image,
                    dtype=np.float64,
                )
                / 255.0
            )


        channel_sum += arr.sum(
            axis=(0, 1)
        )

        channel_sq_sum += (
            np.square(arr)
            .sum(
                axis=(0, 1)
            )
        )

        pixel_count += (
            arr.shape[0]
            * arr.shape[1]
        )


    mean = (
        channel_sum
        / pixel_count
    )

    variance = (
        channel_sq_sum
        / pixel_count
        - mean**2
    )

    std = np.sqrt(
        np.maximum(
            variance,
            0.0,
        )
    )


    return (
        mean,
        std,
        pixel_count,
        len(df),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--train-manifest",
        required=True,
    )

    args = parser.parse_args()


    (
        mean,
        std,
        pixel_count,
        image_count,
    ) = compute_stats(
        args.dataset_root,
        args.train_manifest,
    )


    print()
    print(
        "Images:",
        image_count,
    )

    print(
        "Mean:",
        mean.tolist(),
    )

    print(
        "Std:",
        std.tolist(),
    )

    print(
        "Pixel count:",
        pixel_count,
    )


if __name__ == "__main__":
    main()