import cv2
import numpy as np
import tensorflow as tf

from src.data_pipeline import CLASS_NAMES

LAST_CONV_LAYER_NAME = "out_relu"

#: Names of the custom classification-head layers, in the exact order they
#: must be applied to the backbone's output. These names **must** match the
#: ``name=`` arguments used when the layers were created in
#: ``src/model.build_model``.
HEAD_LAYER_NAMES = [
    "global_avg_pool",
    "dense_head",
    "head_batchnorm",
    "head_dropout",
    "disease_probabilities",
]


def find_backbone_model(model, name_hint="mobilenet"):
   
    for layer in model.layers:
        if name_hint in layer.name.lower() and isinstance(layer, tf.keras.Model):
            return layer

    available = [layer.name for layer in model.layers]
    raise ValueError(
        f"Could not find a backbone layer whose name contains "
        f"'{name_hint}' among the model's top-level layers. "
        f"Available top-level layer names: {available}"
    )


def _apply_head(model, base_output, head_layer_names=HEAD_LAYER_NAMES):
   
    x = base_output
    for layer_name in head_layer_names:
        layer = model.get_layer(layer_name)
        x = layer(x, training=False)
    return x


def make_gradcam_heatmap(
    img_array, model, base_model, class_index=None, head_layer_names=HEAD_LAYER_NAMES, eps=1e-8
):

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Run the backbone directly (NOT via a reconstructed sub-model) to
        # get the final 7x7x1280 feature map.
        conv_outputs = base_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        # Replay the classification head to get the full softmax vector,
        # shape (1, NUM_CLASSES).
        predictions = _apply_head(model, conv_outputs, head_layer_names)

        # Choose which class to explain. By default we explain the winning
        # class -- the one with the highest predicted probability -- so the
        # heatmap localizes the detected pathology. `tf.argmax` is computed
        # inside the tape on the (constant w.r.t. the tape) prediction, then
        # used only to index; the differentiated quantity is `class_score`.
        if class_index is None:
            target_index = int(tf.argmax(predictions[0]).numpy())
        else:
            target_index = int(class_index)

        # Scalar score for the target class, shape (1,). This is the quantity
        # we differentiate the feature map with respect to.
        class_score = predictions[:, target_index]

    # Gradient of the target class score w.r.t. every element of the
    # 7x7x1280 feature map, shape (1, 7, 7, 1280).
    grads = tape.gradient(class_score, conv_outputs)

    # Global-average-pool the gradients over the spatial (height, width)
    # dimensions to get one importance weight per channel, shape (1280,).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Drop the batch dimension from the feature map: (7, 7, 1280).
    conv_outputs = conv_outputs[0]

    # Weighted sum over the channel dimension: (7, 7, 1280) @ (1280, 1) -> (7, 7, 1)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Keep only regions with a net-positive influence on the target class
    # score (standard Grad-CAM convention), then normalize to [0, 1] for
    # visualization.
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + eps)

    probabilities = predictions[0].numpy().astype("float32")
    return heatmap.numpy().astype("float32"), probabilities, target_index


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    
    height, width = original_image.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # OpenCV colormaps produce BGR images; convert to RGB to match the rest
    # of the pipeline (PIL / Streamlit / matplotlib all expect RGB).
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    original_uint8 = original_image.astype(np.uint8)
    if original_uint8.ndim == 2:
        original_uint8 = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2RGB)

    superimposed_rgb = cv2.addWeighted(
        original_uint8, 1.0 - alpha, heatmap_color_rgb, alpha, 0
    )

    return superimposed_rgb, heatmap_uint8


def get_affected_region_summary(heatmap, threshold=0.5):
   
    height, width = heatmap.shape

    row_edges = [0, height // 3, 2 * height // 3, height]
    col_edges = [0, width // 3, 2 * width // 3, width]

    row_labels = ["upper", "mid", "lower"]
    col_labels = ["left", "center", "right"]

    regions = []
    for i in range(3):
        for j in range(3):
            cell = heatmap[row_edges[i]:row_edges[i + 1], col_edges[j]:col_edges[j + 1]]
            if cell.size == 0:
                continue
            if float(np.max(cell)) >= threshold:
                regions.append(f"{row_labels[i]}-{col_labels[j]} lung field")

    max_activation = float(np.max(heatmap))

    return regions, max_activation


def generate_explanation(
    img_array,
    original_image,
    model,
    base_model,
    class_names=CLASS_NAMES,
    head_layer_names=HEAD_LAYER_NAMES,
):

    # class_index=None -> explain the winning (argmax) class.
    heatmap, probabilities, predicted_index = make_gradcam_heatmap(
        img_array, model, base_model, class_index=None, head_layer_names=head_layer_names
    )

    overlay, _ = overlay_heatmap_on_image(original_image, heatmap)

    affected_regions, max_activation = get_affected_region_summary(heatmap)

    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    class_probabilities = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "probabilities": probabilities,
        "heatmap": heatmap,
        "overlay": overlay,
        "affected_regions": affected_regions,
        "max_activation": max_activation,
    }
