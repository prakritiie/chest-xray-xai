"""
smoke_test.py
=============

Lightweight end-to-end smoke test for the Chest X-ray multi-class Explainable
AI project.

This script is **not** part of the production pipeline. It exists to verify,
in well under a minute and using only synthetic data, that every module is
correctly wired together:

1. ``build_model()`` constructs a model with the exact layer names that
   ``src/explainability.py`` depends on.
2. ``load_and_preprocess_single_image()`` correctly handles a raw uploaded
   image (including a grayscale image).
3. ``generate_explanation()`` (Grad-CAM) runs end-to-end on a freshly-built
   model and produces correctly-shaped, valid outputs.
4. The model can be saved to ``.h5`` and reloaded with
   ``tf.keras.models.load_model()`` -- and Grad-CAM still works on the
   reloaded model, producing the *same* prediction and heatmap as before
   the round-trip. This is the critical check that the whole
   "call base_model directly inside a GradientTape" explainability
   approach survives a real save/load cycle.
5. ``unfreeze_top_layers()`` produces a model that still runs a forward pass.
6. ``get_datasets()`` and ``compute_class_weights()`` work correctly against
   a tiny synthetic directory tree matching the expected dataset layout.
7. ``evaluate.py``'s ``get_predictions()`` and ``print_metrics_report()`` run
   end-to-end against the synthetic test set and the reloaded model.

Run with::

    python smoke_test.py

A clean exit with a final "ALL SMOKE TESTS PASSED" message means the core
pipeline is wired together correctly. All test artifacts are created inside
a temporary directory that is removed afterwards.

Note on weights: every model in this script is built with ``weights=None``
(random initialization) rather than ``weights="imagenet"``. This is
intentional -- it lets the smoke test run instantly and without any network
access. It has no bearing on the *correctness* of the checks above, all of
which concern wiring and shapes, not learned weight quality.
"""

import os

# Reduce TensorFlow's startup log verbosity (INFO/WARNING) for cleaner test
# output. Must be set before TensorFlow is imported (directly or via `src`).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import shutil
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

from evaluate import get_predictions, print_metrics_report
from src.data_pipeline import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_class_weights,
    get_datasets,
    load_and_preprocess_single_image,
)
from src.explainability import (
    HEAD_LAYER_NAMES,
    LAST_CONV_LAYER_NAME,
    find_backbone_model,
    generate_explanation,
)
from src.model import build_model, unfreeze_top_layers


def _section(title):
    """Prints a visually distinct section header to the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_build_model():
    """Step 1: build_model() produces the expected architecture."""
    _section("1. build_model()")
    model, base_model = build_model(weights=None)

    layer_names = [layer.name for layer in model.layers]
    print(f"Top-level layer names: {layer_names}")

    for name in HEAD_LAYER_NAMES:
        assert name in layer_names, f"Expected head layer '{name}' not found"
    assert base_model.name in layer_names, "base_model not found among top-level layers"
    assert model.output_shape == (None, NUM_CLASSES), f"Unexpected output shape: {model.output_shape}"
    assert base_model.get_layer(LAST_CONV_LAYER_NAME) is not None

    print(f"base_model name: '{base_model.name}'")
    print(f"model output shape: {model.output_shape}")
    print("OK")
    return model, base_model


def check_single_image_preprocessing():
    """Step 2: load_and_preprocess_single_image() handles RGB and grayscale input."""
    _section("2. load_and_preprocess_single_image()")

    # Arbitrary-resolution RGB image.
    raw_rgb = (np.random.rand(150, 100, 3) * 255).astype(np.uint8)
    preprocessed, resized = load_and_preprocess_single_image(raw_rgb)
    print(f"RGB input  -> preprocessed {preprocessed.shape} {preprocessed.dtype}, "
          f"resized {resized.shape} {resized.dtype}")
    assert preprocessed.shape == (1, 224, 224, 3)
    assert preprocessed.dtype == np.float32
    assert preprocessed.min() >= -1.0 and preprocessed.max() <= 1.0
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8

    # 2D grayscale image -- must be converted to 3-channel RGB.
    raw_gray = (np.random.rand(80, 64) * 255).astype(np.uint8)
    preprocessed_gray, resized_gray = load_and_preprocess_single_image(raw_gray)
    print(f"Gray input -> preprocessed {preprocessed_gray.shape} {preprocessed_gray.dtype}, "
          f"resized {resized_gray.shape} {resized_gray.dtype}")
    assert preprocessed_gray.shape == (1, 224, 224, 3)
    assert resized_gray.shape == (224, 224, 3)

    print("OK")
    return preprocessed, resized


def check_gradcam(model, base_model, preprocessed, resized, step_num, label):
    """Steps 3 / 5: generate_explanation() produces valid, correctly-shaped output."""
    _section(f"{step_num}. generate_explanation() [{label}]")

    explanation = generate_explanation(preprocessed, resized, model, base_model)

    print(f"predicted_class:       {explanation['predicted_class']}")
    print(f"predicted_index:       {explanation['predicted_index']}")
    print(f"confidence:            {explanation['confidence']:.6f}")
    print(f"class_probabilities:   {explanation['class_probabilities']}")
    print(f"probabilities:         shape={explanation['probabilities'].shape}, "
          f"sum={explanation['probabilities'].sum():.6f}")
    print(f"heatmap:               shape={explanation['heatmap'].shape}, "
          f"dtype={explanation['heatmap'].dtype}, "
          f"range=[{explanation['heatmap'].min():.4f}, {explanation['heatmap'].max():.4f}]")
    print(f"overlay:               shape={explanation['overlay'].shape}, "
          f"dtype={explanation['overlay'].dtype}")
    print(f"affected_regions:      {explanation['affected_regions']}")
    print(f"max_activation:        {explanation['max_activation']:.6f}")

    assert explanation["heatmap"].shape == (7, 7)
    assert explanation["heatmap"].dtype == np.float32
    assert explanation["heatmap"].min() >= 0.0 and explanation["heatmap"].max() <= 1.0
    assert explanation["overlay"].shape == resized.shape
    assert explanation["overlay"].dtype == np.uint8
    assert explanation["predicted_class"] in CLASS_NAMES
    assert 0 <= explanation["predicted_index"] < NUM_CLASSES
    assert 0.0 <= explanation["confidence"] <= 1.0
    assert explanation["probabilities"].shape == (NUM_CLASSES,)
    # Softmax probabilities must sum to 1 and the dict must cover every class.
    assert abs(float(explanation["probabilities"].sum()) - 1.0) < 1e-4
    assert set(explanation["class_probabilities"].keys()) == set(CLASS_NAMES)
    # The winning class must be consistent with the probability vector.
    assert explanation["predicted_class"] == CLASS_NAMES[int(np.argmax(explanation["probabilities"]))]
    assert isinstance(explanation["affected_regions"], list)
    assert len(explanation["affected_regions"]) <= 9

    print("OK")
    return explanation


def check_save_and_reload(model, tmp_dir):
    """Step 4: model.save() + tf.keras.models.load_model() round-trip."""
    _section("4. model.save() + tf.keras.models.load_model() round-trip")

    save_path = os.path.join(tmp_dir, "smoke_test_model.h5")
    model.save(save_path)
    print(f"Saved model to '{save_path}' ({os.path.getsize(save_path) / 1e6:.1f} MB)")

    reloaded = tf.keras.models.load_model(save_path)
    print("Reloaded model with tf.keras.models.load_model().")

    reloaded_layer_names = [layer.name for layer in reloaded.layers]
    for name in HEAD_LAYER_NAMES:
        assert name in reloaded_layer_names, f"Expected head layer '{name}' missing after reload"
    assert reloaded.output_shape == (None, NUM_CLASSES)

    reloaded_base_model = find_backbone_model(reloaded)
    print(f"find_backbone_model() -> '{reloaded_base_model.name}'")

    # The Grad-CAM tap point must still exist inside the reloaded backbone.
    out_relu = reloaded_base_model.get_layer(LAST_CONV_LAYER_NAME)
    assert out_relu is not None

    print("OK")
    return reloaded, reloaded_base_model


def check_consistency(explanation_fresh, explanation_reloaded):
    """Step 6: fresh and reloaded models must agree on prediction + heatmap."""
    _section("6. Consistency check: fresh model vs. reloaded model")

    prob_fresh = explanation_fresh["probabilities"]
    prob_reloaded = explanation_reloaded["probabilities"]
    prob_diff = float(np.max(np.abs(prob_fresh - prob_reloaded)))
    print(f"max |Δ probability|: {prob_diff:.2e}")
    print(f"fresh probs:    {np.array2string(prob_fresh, precision=6)}")
    print(f"reloaded probs: {np.array2string(prob_reloaded, precision=6)}")
    assert prob_diff < 1e-4, "Save/reload changed the model's prediction"
    assert explanation_fresh["predicted_index"] == explanation_reloaded["predicted_index"], (
        "Save/reload changed the predicted class"
    )

    heatmap_close = np.allclose(
        explanation_fresh["heatmap"], explanation_reloaded["heatmap"], atol=1e-4
    )
    print(f"heatmaps match within tolerance: {heatmap_close}")
    assert heatmap_close, "Save/reload changed the Grad-CAM heatmap"

    print("OK")


def check_unfreeze_top_layers():
    """Step 7: unfreeze_top_layers() produces a still-usable model."""
    _section("7. unfreeze_top_layers()")

    model, base_model = build_model(weights=None)
    assert base_model.trainable is False

    model = unfreeze_top_layers(model, base_model, num_layers=10, learning_rate=1e-5)
    assert base_model.trainable is True

    trainable_backbone_layers = [layer for layer in base_model.layers if layer.trainable]
    frozen_backbone_layers = [layer for layer in base_model.layers if not layer.trainable]
    print(
        f"Backbone layers: {len(base_model.layers)} total, "
        f"{len(trainable_backbone_layers)} trainable, "
        f"{len(frozen_backbone_layers)} frozen"
    )
    assert len(trainable_backbone_layers) > 0
    assert len(frozen_backbone_layers) > 0

    # The recompiled model must still run a forward pass.
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (1, NUM_CLASSES)

    print("OK")


def build_dummy_dataset(root_dir, counts):
    """Creates a tiny synthetic dataset directory tree of random JPEG images.

    Parameters
    ----------
    root_dir : str
        Root directory in which ``train/``, ``val/``, and ``test/``
        subdirectories are created.
    counts : dict
        Nested mapping ``counts[split][class_name] = n_images``, e.g.
        ``{"train": {"COVID19": 2, "NORMAL": 2, "PNEUMONIA": 4,
        "TURBERCULOSIS": 2}, ...}``.
    """
    rng = np.random.default_rng(42)
    for split, class_counts in counts.items():
        for class_name, n in class_counts.items():
            class_dir = os.path.join(root_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(class_dir, f"img_{i}.jpeg"), "JPEG")


def check_data_pipeline(tmp_dir):
    """Step 8: get_datasets() and compute_class_weights() on synthetic data."""
    _section("8. get_datasets() + compute_class_weights()")

    data_dir = os.path.join(tmp_dir, "dummy_chest_xray")
    # COVID19=2, NORMAL=2, PNEUMONIA=4, TURBERCULOSIS=2  (total train = 10)
    counts = {
        "train": {"COVID19": 2, "NORMAL": 2, "PNEUMONIA": 4, "TURBERCULOSIS": 2},
        "val": {"COVID19": 1, "NORMAL": 1, "PNEUMONIA": 1, "TURBERCULOSIS": 1},
        "test": {"COVID19": 1, "NORMAL": 1, "PNEUMONIA": 1, "TURBERCULOSIS": 1},
    }
    build_dummy_dataset(data_dir, counts)

    train_ds, val_ds, test_ds, class_names = get_datasets(data_dir=data_dir, batch_size=4)
    print(f"class_names: {class_names}")
    assert class_names == CLASS_NAMES

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        images, labels = next(iter(ds))
        print(f"{name}: images {tuple(images.shape)} {images.dtype}, "
              f"labels {tuple(labels.shape)} {labels.dtype}")
        assert images.shape[1:] == (224, 224, 3)
        assert images.dtype == tf.float32
        # Labels are one-hot for the 4-class problem.
        assert labels.shape[1:] == (NUM_CLASSES,)

    class_weights = compute_class_weights(data_dir=data_dir)
    print(f"class_weights: {class_weights}")
    # train counts: COVID19=2, NORMAL=2, PNEUMONIA=4, TURBERCULOSIS=2; total=10; n_classes=4
    # weight[i] = total / (n_classes * count_i)
    #   COVID19/NORMAL/TB: 10 / (4 * 2) = 1.25 ; PNEUMONIA: 10 / (4 * 4) = 0.625
    assert set(class_weights.keys()) == set(range(NUM_CLASSES))
    assert abs(class_weights[0] - 1.25) < 1e-6   # COVID19
    assert abs(class_weights[1] - 1.25) < 1e-6   # NORMAL
    assert abs(class_weights[2] - 0.625) < 1e-6  # PNEUMONIA
    assert abs(class_weights[3] - 1.25) < 1e-6   # TUBERCULOSIS

    print("OK")
    return test_ds


def check_evaluate_pipeline(reloaded_model, test_ds):
    """Step 9: evaluate.py's get_predictions() + print_metrics_report()."""
    _section("9. evaluate.py: get_predictions() + print_metrics_report()")

    y_true, y_prob = get_predictions(reloaded_model, test_ds)
    print(f"y_true: {y_true}")
    print(f"y_prob shape: {y_prob.shape}")
    # y_true is 1-D class indices; y_prob is 2-D (n_images, NUM_CLASSES).
    assert y_true.ndim == 1
    assert y_prob.shape == (y_true.shape[0], NUM_CLASSES)
    assert y_true.shape[0] == 4  # 1 image per class in the dummy test set
    # Each row is a softmax distribution summing to ~1.
    assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-4)

    y_pred = np.argmax(y_prob, axis=1)
    metrics = print_metrics_report(y_true, y_pred, y_prob, CLASS_NAMES)

    for key in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc_macro",
        "roc_auc_weighted",
        "confusion_matrix",
        "false_negatives",
    ):
        assert key in metrics, f"Missing key '{key}' in metrics dict"

    # The confusion matrix must be NUM_CLASSES x NUM_CLASSES.
    assert metrics["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)

    print(f"\nmetrics dict keys: {list(metrics.keys())}")
    print("OK")


def main():
    tmp_dir = tempfile.mkdtemp(prefix="chest_xray_smoke_")
    print(f"Using temporary directory: {tmp_dir}")

    try:
        model, base_model = check_build_model()
        preprocessed, resized = check_single_image_preprocessing()

        explanation_fresh = check_gradcam(
            model, base_model, preprocessed, resized, step_num=3, label="freshly-built model"
        )

        reloaded_model, reloaded_base_model = check_save_and_reload(model, tmp_dir)

        explanation_reloaded = check_gradcam(
            reloaded_model, reloaded_base_model, preprocessed, resized,
            step_num=5, label="reloaded model",
        )

        check_consistency(explanation_fresh, explanation_reloaded)
        check_unfreeze_top_layers()

        test_ds = check_data_pipeline(tmp_dir)
        check_evaluate_pipeline(reloaded_model, test_ds)

        _section("ALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {tmp_dir}")


if __name__ == "__main__":
    main()
