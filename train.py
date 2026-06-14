"""
train.py
========

End-to-end training script for the Chest X-ray multi-class Explainable AI
project (4 classes: COVID19, NORMAL, PNEUMONIA, TURBERCULOSIS).

This script orchestrates the full training workflow:

1. Loads the train / validation / test ``tf.data.Dataset`` pipelines and
   computes class weights (``src.data_pipeline``).
2. Builds the MobileNetV2-based model with a frozen backbone
   (``src.model.build_model``).
3. **Stage 1**: trains the classification head against the frozen backbone
   for ``--epochs`` epochs, using ``ModelCheckpoint``, ``EarlyStopping``
   (monitoring validation AUC, with ``restore_best_weights=True``), and
   ``ReduceLROnPlateau``.
4. Saves a full-model snapshot after Stage 1 (``models/stage1_full_model.h5``)
   as a safety fallback.
5. **Stage 2 (optional)**: if ``--fine_tune_epochs > 0``, unfreezes the top
   ``--unfreeze_layers`` layers of the backbone
   (``src.model.unfreeze_top_layers``) and continues training at a much
   lower learning rate, with a *fresh* set of callbacks. If Stage 2 does not
   improve validation AUC over Stage 1, the Stage 1 snapshot is reloaded so
   the final model is never worse than Stage 1.
6. Saves the final model to ``models/<--model_name>`` (``.h5`` format),
   evaluates it on the test set, and writes a training-history plot to
   ``outputs/training_history.png``.

Usage
-----

Stage 1 only (head-only training, the default)::

    python train.py --data_dir data/chest_xray --epochs 15

Stage 1 followed by Stage 2 fine-tuning::

    python train.py --data_dir data/chest_xray --epochs 15 --fine_tune_epochs 10

Run ``python train.py --help`` for the full list of options.

Important implementation note on callbacks
--------------------------------------------
``ModelCheckpoint`` and ``EarlyStopping`` track their "best so far" value
internally as state on the callback *instance*. If the same callback
instances were reused across both training stages, that internal "best"
state would carry over in subtly incorrect ways (e.g. Stage 2's
``EarlyStopping`` would compare against Stage 1's best AUC using a
*different* model configuration). To avoid this, :func:`build_callbacks`
constructs a brand-new set of callback instances for each stage.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # Headless backend: this script may run without a display.
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_pipeline import compute_class_weights, get_datasets
from src.model import build_model, unfreeze_top_layers


def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train the MobileNetV2-based chest X-ray multi-class disease classifier."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/chest_xray",
        help="Root directory of the dataset, containing train/val/test subfolders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs for Stage 1 (frozen-backbone) training.",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs for optional Stage 2 fine-tuning. If 0 "
            "(the default), Stage 2 is skipped entirely."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training, validation, and test datasets.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for Stage 1 (frozen-backbone) training.",
    )
    parser.add_argument(
        "--fine_tune_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for Stage 2 fine-tuning.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout rate applied in the classification head.",
    )
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=30,
        help="Number of backbone layers (from the end) to unfreeze during Stage 2.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory where model checkpoints and the final model are saved.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where training logs and plots are saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="chest_xray_mobilenetv2.h5",
        help="Filename for the final saved model, written inside --model_dir.",
    )

    return parser.parse_args()


def build_callbacks(weights_checkpoint_path, log_path):
    """Builds a *fresh* list of training callbacks for one training stage.

    Parameters
    ----------
    weights_checkpoint_path : str
        Path (must end in ``.weights.h5``) where the best-weights-so-far
        checkpoint for this stage is saved.
    log_path : str
        Path to a CSV file where per-epoch metrics for this stage are
        logged.

    Returns
    -------
    list
        A list of newly-constructed Keras callbacks:

        * ``ModelCheckpoint``: saves only the model's weights (not the full
          model) whenever validation AUC improves. This acts as an
          on-disk safety net independent of the in-memory
          ``restore_best_weights`` behavior of ``EarlyStopping`` below.
        * ``EarlyStopping``: monitors validation AUC, stops training if it
          does not improve for ``patience`` epochs, and -- critically --
          restores the model's in-memory weights to those from the best
          epoch at the end of ``fit()``, regardless of whether early
          stopping actually triggered.
        * ``ReduceLROnPlateau``: halves the learning rate if validation loss
          plateaus for 2 epochs, helping the model escape shallow local
          minima late in training.
        * ``CSVLogger``: writes per-epoch metrics to ``log_path`` for later
          inspection.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(log_path),
    ]


def plot_history(history_list, save_path):
    """Plots loss / accuracy / AUC curves across one or more training stages.

    Produces a row of subplots (loss, accuracy, AUC), each showing both the
    training and validation curves across the *entire* training run. If
    ``history_list`` contains more than one stage's history (i.e. Stage 2
    fine-tuning was performed), a vertical dashed line marks the epoch at
    which Stage 2 began.

    Parameters
    ----------
    history_list : list of dict
        A list of ``history.history`` dictionaries (as produced by
        ``model.fit(...).history``), one per training stage, in
        chronological order.
    save_path : str
        File path (e.g. ``outputs/training_history.png``) to save the
        resulting figure to.
    """
    metrics = ["loss", "accuracy", "auc"]
    titles = ["Loss", "Accuracy", "AUC (macro, multi-label)"]

    # Concatenate each metric's per-epoch values across all stages.
    combined = {}
    for stage_history in history_list:
        for key, values in stage_history.items():
            combined.setdefault(key, []).extend(values)

    # If there were multiple stages, mark the boundary between them.
    stage_boundary = None
    if len(history_list) > 1:
        stage_boundary = len(history_list[0]["loss"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.ravel()

    for ax, metric, title in zip(axes, metrics, titles):
        val_key = f"val_{metric}"

        if metric not in combined:
            ax.set_visible(False)
            continue

        epochs_range = range(1, len(combined[metric]) + 1)
        ax.plot(epochs_range, combined[metric], label=f"Train {title}", color="#1f77b4")

        if val_key in combined:
            ax.plot(epochs_range, combined[val_key], label=f"Val {title}", color="#ff7f0e")

        if stage_boundary is not None:
            ax.axvline(
                x=stage_boundary + 0.5,
                color="gray",
                linestyle="--",
                linewidth=1,
                label="Fine-tuning starts",
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Training History", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _section(title):
    """Prints a visually distinct section header to the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Runs the full training pipeline based on parsed command-line arguments."""
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Data
    # -----------------------------------------------------------------
    _section("Loading datasets")
    train_ds, val_ds, test_ds, class_names = get_datasets(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    class_weights = compute_class_weights(data_dir=args.data_dir)
    print(f"Class names: {class_names}")
    print(f"Class weights (to counteract training-set imbalance): {class_weights}")

    # -----------------------------------------------------------------
    # 2. Model
    # -----------------------------------------------------------------
    _section("Building model (Stage 1: frozen MobileNetV2 backbone)")
    model, base_model = build_model(
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        freeze_backbone=True,
    )
    model.summary()

    history_list = []

    # -----------------------------------------------------------------
    # 3. Stage 1: train the classification head
    # -----------------------------------------------------------------
    _section(f"Stage 1: training head for up to {args.epochs} epochs")

    stage1_weights_path = os.path.join(args.model_dir, "stage1_best.weights.h5")
    stage1_log_path = os.path.join(args.output_dir, "stage1_training_log.csv")
    stage1_callbacks = build_callbacks(stage1_weights_path, stage1_log_path)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
    )
    history_list.append(history1.history)

    best_val_auc = max(history1.history["val_auc"])
    print(f"\nStage 1 complete. Best validation AUC: {best_val_auc:.4f}")

    # Save a full-model snapshot of Stage 1 as a safety fallback in case
    # Stage 2 fine-tuning makes things worse.
    stage1_full_path = os.path.join(args.model_dir, "stage1_full_model.h5")
    model.save(stage1_full_path)
    print(f"Saved Stage 1 full-model snapshot to '{stage1_full_path}'")

    # -----------------------------------------------------------------
    # 4. Stage 2 (optional): fine-tune the top of the backbone
    # -----------------------------------------------------------------
    if args.fine_tune_epochs > 0:
        _section(
            f"Stage 2: fine-tuning top {args.unfreeze_layers} backbone layers "
            f"for up to {args.fine_tune_epochs} epochs"
        )

        model = unfreeze_top_layers(
            model,
            base_model,
            num_layers=args.unfreeze_layers,
            learning_rate=args.fine_tune_learning_rate,
        )
        model.summary()

        stage2_weights_path = os.path.join(args.model_dir, "stage2_best.weights.h5")
        stage2_log_path = os.path.join(args.output_dir, "stage2_training_log.csv")
        stage2_callbacks = build_callbacks(stage2_weights_path, stage2_log_path)

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=stage2_callbacks,
        )
        history_list.append(history2.history)

        fine_tune_best_val_auc = max(history2.history["val_auc"])
        print(f"\nStage 2 complete. Best validation AUC: {fine_tune_best_val_auc:.4f}")

        if fine_tune_best_val_auc <= best_val_auc:
            print(
                "Fine-tuning did NOT improve validation AUC "
                f"({fine_tune_best_val_auc:.4f} <= {best_val_auc:.4f}). "
                f"Reverting to the Stage 1 model snapshot from "
                f"'{stage1_full_path}'."
            )
            model = tf.keras.models.load_model(stage1_full_path)
        else:
            print(
                "Fine-tuning improved validation AUC "
                f"({fine_tune_best_val_auc:.4f} > {best_val_auc:.4f}). "
                "Keeping the fine-tuned model."
            )
            best_val_auc = fine_tune_best_val_auc
    else:
        print("\nSkipping Stage 2 fine-tuning (--fine_tune_epochs is 0).")

    # -----------------------------------------------------------------
    # 5. Save final model, evaluate, and plot training history
    # -----------------------------------------------------------------
    _section("Saving final model and evaluating on the test set")

    final_path = os.path.join(args.model_dir, args.model_name)
    model.save(final_path)
    print(f"Saved final model to '{final_path}'")
    print(f"Best overall validation AUC: {best_val_auc:.4f}")

    test_results = model.evaluate(test_ds, return_dict=True)
    print("\nTest set results:")
    for metric_name, value in test_results.items():
        print(f"  {metric_name}: {value:.4f}")

    history_plot_path = os.path.join(args.output_dir, "training_history.png")
    plot_history(history_list, history_plot_path)
    print(f"\nSaved training history plot to '{history_plot_path}'")


if __name__ == "__main__":
    main()
