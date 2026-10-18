from copy import deepcopy


THESIS_BACKBONE_VARIANTS = {

    "r18_none": {
        "convnet_type":
            "remote_resnet18",

        "attention_type":
            "none",
    },

    "r18_eca": {
        "convnet_type":
            "remote_resnet18",

        "attention_type":
            "eca",

        "eca_gamma":
            2.0,

        "eca_b":
            1.0,
    },

    "r18_cbam": {
        "convnet_type":
            "remote_resnet18",

        "attention_type":
            "cbam",

        "cbam_reduction":
            16,

        "cbam_spatial_kernel":
            7,
    },

    "r18_procbam": {
        "convnet_type":
            "remote_resnet18",

        "attention_type":
            "procbam",

        "procbam_reduction":
            16,

        "procbam_spatial_kernel":
            7,

        "procbam_se_reduction":
            16,
    },

    "r34_none": {
        "convnet_type":
            "remote_resnet34",

        "attention_type":
            "none",
    },

    "r34_eca": {
        "convnet_type":
            "remote_resnet34",

        "attention_type":
            "eca",

        "eca_gamma":
            2.0,

        "eca_b":
            1.0,
    },

    "r34_cbam": {
        "convnet_type":
            "remote_resnet34",

        "attention_type":
            "cbam",

        "cbam_reduction":
            16,

        "cbam_spatial_kernel":
            7,
    },

    "r34_procbam": {
        "convnet_type":
            "remote_resnet34",

        "attention_type":
            "procbam",

        "procbam_reduction":
            16,

        "procbam_spatial_kernel":
            7,

        "procbam_se_reduction":
            16,
    },
}


def get_backbone_variant(
    variant_name,
):
    """
    Mengembalikan copy konfigurasi agar dictionary
    registry tidak termodifikasi oleh caller.
    """

    variant_name = (
        str(variant_name)
        .lower()
    )

    if (
        variant_name
        not in THESIS_BACKBONE_VARIANTS
    ):
        available = ", ".join(
            THESIS_BACKBONE_VARIANTS.keys()
        )

        raise ValueError(
            f"Unknown backbone variant: "
            f"{variant_name}. "
            f"Available: {available}"
        )

    return deepcopy(
        THESIS_BACKBONE_VARIANTS[
            variant_name
        ]
    )


def get_all_backbone_variants():
    return {
        name: deepcopy(config)
        for name, config
        in THESIS_BACKBONE_VARIANTS.items()
    }