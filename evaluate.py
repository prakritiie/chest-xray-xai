import argparse
import os

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.data_pipeline import CLASS_NAMES, NUM_CLASSES, get_datasets


def parse_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-class chest X-ray classifier on the held-out test set."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/chest_xray",
        help="Root directory of the dataset, containing train/val/test subfolders.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/chest_xray_mobilenetv2.h5",
        help="Path to the trained Keras model file (.h5), as saved by train.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where evaluation plots (confusion matrix, ROC curves) are saved.",
    )

    return parser.parse_args()


def get_predictions(model, test_ds):

    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        # Labels are one-hot (label_mode="categorical"); convert to indices.
        y_true.append(np.argmax(labels.numpy(), axis=1))

    return np.concatenate(y_true), np.concatenate(y_prob, axis=0)


def print_metrics_report(y_true, y_pred, y_prob, class_names=CLASS_NAMES):

    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Multi-class One-vs-Rest ROC-AUC. This requires every class to appear in
    # y_true; on a degenerate/tiny test set it can raise, so guard it.
    roc_auc_macro = float("nan")
    roc_auc_weighted = float("nan")
    try:
        roc_auc_macro = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=labels
        )
        roc_auc_weighted = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="weighted", labels=labels
        )
    except ValueError:
        pass

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    false_negatives = {
        class_names[i]: int(cm[i].sum() - cm[i, i]) for i in range(len(class_names))
    }

    _section("TEST SET PERFORMANCE METRICS (MULTI-CLASS)")
    print(f"{'Overall Accuracy:':<26}{accuracy:.4f}\n")

    print(f"{'':<16}{'Macro':>12}{'Weighted':>12}")
    print(f"{'Precision':<16}{precision_macro:>12.4f}{precision_weighted:>12.4f}")
    print(f"{'Recall':<16}{recall_macro:>12.4f}{recall_weighted:>12.4f}")
    print(f"{'F1-Score':<16}{f1_macro:>12.4f}{f1_weighted:>12.4f}")
    if np.isnan(roc_auc_macro):
        print(f"{'ROC-AUC (OvR)':<16}{'N/A':>12}{'N/A':>12}  (a class is missing from test labels)")
    else:
        print(f"{'ROC-AUC (OvR)':<16}{roc_auc_macro:>12.4f}{roc_auc_weighted:>12.4f}")

    print("\nPer-class metrics:")
    print(f"{'Class':<16}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
    for i, name in enumerate(class_names):
        print(
            f"{name:<16}{per_class_precision[i]:>12.4f}"
            f"{per_class_recall[i]:>12.4f}{per_class_f1[i]:>12.4f}"
        )

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    header = "actual \\ pred  " + "".join(f"{name[:10]:>12}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>12d}" for j in range(len(class_names)))
        print(f"{name:<14}{row}")

    print("\nFull per-class report:")
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
        )
    )

    _section("WHY RECALL IS THE PRIORITY METRIC FOR THIS APPLICATION")
    fn_lines = "\n".join(
        f"      - {name:<14}: {count} missed case(s)"
        for name, count in false_negatives.items()
    )
    print(f"""
This is a multi-pathology *screening* tool that must distinguish four
conditions: COVID19, NORMAL, PNEUMONIA, and TURBERCULOSIS. The most
dangerous error is a FALSE NEGATIVE for any of the disease classes -- the
model labels a sick patient as NORMAL (or as the wrong disease), so the true
condition goes untreated.

Per-class False Negatives on this test set (true class -> predicted as
something else):
{fn_lines}


  * MACRO-averaged Recall treats every class equally, regardless of how many
    test images it has. This is the number to watch for clinical safety: it
    will drop sharply if the model is failing on a RARE-but-critical class
    such as TURBERCULOSIS (the smallest class in this dataset), even if
    overall accuracy still looks high.

  * WEIGHTED-averaged Recall weights each class by its support (number of
    true images), so it tracks closer to overall accuracy and is dominated
    by the common classes.

A FALSE POSITIVE (flagging a healthy patient, or confusing two diseases) is
not harmless here -- it can route a patient to the wrong follow-up -- but it
is still far less dangerous than telling a TB or COVID patient they are
healthy. This is why training:

  1. Applies per-class ``class_weight`` (see ``compute_class_weights``) so
     the rare TURBERCULOSIS class is not drowned out during training.
  2. Monitors ``val_auc`` -- not ``val_accuracy`` -- for ``EarlyStopping``
     and ``ModelCheckpoint``, since on an imbalanced 4-class problem
     accuracy can look high while the model quietly never predicts the
     rarest disease.

If MACRO Recall is much lower than WEIGHTED Recall, that gap is the tell-tale
sign the model is sacrificing a minority disease class -- the first thing to
investigate before trusting the model.
""")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "false_negatives": false_negatives,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix -- Test Set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, class_names, save_path):
  
    labels = list(range(len(class_names)))
    y_true_bin = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, name in enumerate(class_names):
        # A class absent from y_true has an undefined ROC curve; skip it.
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        try:
            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            label = f"{name} (AUC = {auc_i:.3f})"
        except ValueError:
            label = f"{name}"
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=label)

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("One-vs-Rest ROC Curves -- Test Set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _section(title):
    """Prints a visually distinct section header to the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Runs the full evaluation pipeline based on parsed command-line arguments."""
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No model found at '{args.model_path}'. Train a model first with "
            f"`python train.py`, or pass the correct path via --model_path."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    _section("Loading model")
    model = tf.keras.models.load_model(args.model_path)
    print(f"Loaded model from '{args.model_path}'")

    _section("Loading test dataset")
    _, _, test_ds, class_names = get_datasets(data_dir=args.data_dir)
    print(f"Class names: {class_names}")

    _section("Running inference on test set")
    y_true, y_prob = get_predictions(model, test_ds)
    y_pred = np.argmax(y_prob, axis=1)
    print(f"Collected predictions for {len(y_true)} test images across {NUM_CLASSES} classes.")

    metrics = print_metrics_report(y_true, y_pred, y_prob, class_names)

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    print(f"\nSaved confusion matrix plot to '{cm_path}'")

    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_roc_curves(y_true, y_prob, class_names, roc_path)
    print(f"Saved One-vs-Rest ROC curve plot to '{roc_path}'")

    return metrics

if __name__ == "__main__":
    main()
