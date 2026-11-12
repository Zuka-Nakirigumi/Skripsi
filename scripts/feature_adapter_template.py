"""MANDATORY PyCIL-specific adapter template.

Implement this by checking YOUR versions of models/{fetril,fecam,ssre,pass,efc}.py,
convs/, thesis/experiments/runner.py, and the final-task production test loader.

This file deliberately raises NotImplementedError rather than fabricating embeddings.
Do not run batch extraction until *one production checkpoint per method* passes.

Security: load only trusted local PyCIL-produced checkpoints. Do not unpickle
unknown third-party files. Prefer torch.load(..., weights_only=True) for plain
state_dict checkpoints when supported; method_state_*.pkl may contain sklearn
objects and needs separately reviewed trusted restoration.
"""


def extract_from_checkpoint(*, run_dir, checkpoint_path, effective_args,
                            method, protocol, backbone, attention, limit):
    """Return a dict with:
        features: float32 array [N,512] from semantic pre-classifier representation
        y_true: int64 array [N] in canonical prediction-file test order
        sample_id: optional array [N] of stable IDs
        sample_path: optional array [N] of actual evaluation image paths
        checkpoint_loaded: True ONLY after real weights restored
        semantic_eval_features: True ONLY for normal deterministic eval path

    limit: 16 or another integer for smoke, None for all 2000 AID test samples.

    Requirements by method (must be checked in actual PyCIL source):
    - FeTrIL: frozen backbone representation used by LinearSVC.
    - FeCAM: features used by Mahalanobis prototype statistics.
    - SSRE: compressed/merged final representation, NOT intermediate branch.
    - PASS: normal semantic inference, NOT four rotated SSL copies.
    - EFC: final feature path used by multi-head classifier.

    Use model.eval(), torch.inference_mode(), deterministic production eval
    transforms, shuffle=False, no WeightedRandomSampler, and load final task.
    """
    raise NotImplementedError(
        f"Cannot safely restore {method} without verifying the actual "
        f"PyCIL learner/backbone constructor and evaluation loader. "
        f"Review scripts/feature_adapter_template.py before proceeding."
    )
