from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)


def canonical_top1(
    y_pred,
    y_true,
):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    return float(
        np.mean(
            y_pred == y_true
        )
        * 100.0
    )


def task_ranges(
    task_sizes,
):
    ranges = []

    start = 0

    for size in task_sizes:
        end = start + int(size)

        ranges.append(
            (start, end)
        )

        start = end

    return ranges


def task_accuracy_row(
    y_pred,
    y_true,
    task_sizes,
    current_task,
):
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    y_true = np.asarray(y_true)

    ranges = task_ranges(
        task_sizes
    )

    row = np.full(
        len(task_sizes),
        np.nan,
        dtype=np.float64,
    )

    for task_id in range(
        current_task + 1
    ):
        start, end = ranges[task_id]

        mask = (
            (y_true >= start)
            & (y_true < end)
        )

        if not np.any(mask):
            continue

        row[task_id] = (
            np.mean(
                y_pred[mask]
                == y_true[mask]
            )
            * 100.0
        )

    return row


def class_accuracy_row(
    y_pred,
    y_true,
    total_classes,
    max_classes,
):
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    y_true = np.asarray(y_true)

    row = np.full(
        max_classes,
        np.nan,
        dtype=np.float64,
    )

    for class_id in range(
        int(total_classes)
    ):
        mask = (
            y_true == class_id
        )

        if not np.any(mask):
            continue

        row[class_id] = (
            np.mean(
                y_pred[mask]
                == y_true[mask]
            )
            * 100.0
        )

    return row


def macro_f1_seen(
    y_pred,
    y_true,
    total_classes,
):
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    y_true = np.asarray(y_true)

    labels = np.arange(
        int(total_classes)
    )

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )


def confusion_seen(
    y_pred,
    y_true,
    total_classes,
):
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    y_true = np.asarray(y_true)

    labels = np.arange(
        int(total_classes)
    )

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


def compute_aia(
    joint_accuracy_history,
):
    values = np.asarray(
        joint_accuracy_history,
        dtype=np.float64,
    )

    if values.size == 0:
        return None

    return float(
        np.mean(values)
    )


def compute_lta(
    joint_accuracy_history,
):
    if not joint_accuracy_history:
        return None

    return float(
        joint_accuracy_history[-1]
    )


def compute_bwt(
    task_accuracy_matrix,
    completed_tasks,
):
    matrix = np.asarray(
        task_accuracy_matrix,
        dtype=np.float64,
    )

    if completed_tasks <= 1:
        return 0.0

    final_row = (
        completed_tasks - 1
    )

    values = []

    for task_id in range(
        completed_tasks - 1
    ):
        diagonal = matrix[
            task_id,
            task_id,
        ]

        final_accuracy = matrix[
            final_row,
            task_id,
        ]

        if (
            np.isnan(diagonal)
            or np.isnan(final_accuracy)
        ):
            continue

        values.append(
            final_accuracy
            - diagonal
        )

    if not values:
        return None

    return float(
        np.mean(values)
    )


def compute_forgetting_rate(
    class_accuracy_matrix,
    completed_tasks,
):
    matrix = np.asarray(
        class_accuracy_matrix,
        dtype=np.float64,
    )

    if completed_tasks <= 0:
        return None

    final_row = (
        completed_tasks - 1
    )

    values = []

    for class_id in range(
        matrix.shape[1]
    ):
        history = matrix[
            :completed_tasks,
            class_id,
        ]

        valid = history[
            ~np.isnan(history)
        ]

        if len(valid) == 0:
            continue

        final_accuracy = matrix[
            final_row,
            class_id,
        ]

        if np.isnan(
            final_accuracy
        ):
            continue

        forgetting = (
            np.max(valid)
            - final_accuracy
        )

        values.append(
            forgetting
        )

    if not values:
        return None

    return float(
        np.mean(values)
    )


def build_continual_summary(
    joint_accuracy_history,
    macro_f1_history,
    task_accuracy_matrix,
    class_accuracy_matrix,
    completed_tasks,
):
    return {
        "aia":
            compute_aia(
                joint_accuracy_history
            ),

        "last_task_accuracy":
            compute_lta(
                joint_accuracy_history
            ),

        "forgetting_rate":
            compute_forgetting_rate(
                class_accuracy_matrix,
                completed_tasks,
            ),

        "bwt":
            compute_bwt(
                task_accuracy_matrix,
                completed_tasks,
            ),

        "fwt":
            None,

        "fwt_status":
            (
                "not_applicable_under_dynamic_classifier_cil_protocol"
            ),

        "macro_f1_final":
            (
                float(
                    macro_f1_history[-1]
                )
                if macro_f1_history
                else None
            ),

        "macro_f1_history":
            [
                float(value)
                for value
                in macro_f1_history
            ],

        "joint_top1_history":
            [
                float(value)
                for value
                in joint_accuracy_history
            ],
    }
