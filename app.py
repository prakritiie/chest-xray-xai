import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.data_pipeline import load_and_preprocess_single_image
from src.explainability import find_backbone_model, generate_explanation


st.set_page_config(
    page_title="Chest X-ray Multi-Disease | Explainable AI",
    page_icon="🫁",
    layout="wide",
)

MODEL_PATH = "models/chest_xray_mobilenetv2.h5"


@st.cache_resource
def load_model(model_path):

    model = tf.keras.models.load_model(model_path)
    base_model = find_backbone_model(model)
    return model, base_model


def render_header():
    st.title("Explainable AI for Chest X-ray Diagnosis")
    st.markdown(
        "Upload a chest X-ray image to receive an AI-assisted classification "
        "across four classes — **COVID-19, Normal, Pneumonia, and "
        "Tuberculosis** — along with a **Grad-CAM** visual explanation that "
        "highlights the lung regions which most influenced the model's "
        "decision for the predicted disease."
    )
    # st.warning(
    #     "**Disclaimer:** This tool is for **educational and research "
    #     "purposes only**. It is **not** a certified medical device and "
    #     "must **not** be used for real clinical diagnosis, screening, or "
    #     "treatment decisions. Always consult a qualified radiologist or "
    #     "physician for medical advice."
    # )


def render_sidebar():
    with st.sidebar:
        st.header("About this project")
        st.markdown(
            "This app demonstrates an **end-to-end Explainable AI (XAI) "
            "pipeline** for chest X-ray disease screening:\n\n"
            "- **Model:** MobileNetV2 (transfer learning) with a frozen "
            "ImageNet backbone and a custom classification head\n"
            "- **Explainability:** Grad-CAM, computed from the backbone's "
            "final convolutional feature map for the predicted class\n"
            "- **Task:** 4-class classification — COVID19, NORMAL, "
            "PNEUMONIA, TURBERCULOSIS"
        )
        st.markdown("---")


def render_prediction(explanation):

    st.subheader("1. Prediction")

    predicted_class = explanation["predicted_class"]
    confidence = explanation["confidence"]
    class_probabilities = explanation["class_probabilities"]

    # Classes that represent a disease (everything except NORMAL). Used only
    # to choose the alert color: a disease prediction is shown as an error
    # (red), a NORMAL prediction as a success (green).
    if predicted_class == "NORMAL":
        st.success(f"**Prediction: NORMAL** (confidence: {confidence:.1%})")
    else:
        st.error(f"**Prediction: {predicted_class}** (confidence: {confidence:.1%})")

    st.markdown("**Probability across all classes:**")

    # Sort classes by probability (descending) so the most likely diagnosis is
    # at the top. The winning class is bolded.
    ranked = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for name, prob in ranked:
        label = f"**{name}**" if name == predicted_class else name
        col_label, col_bar, col_val = st.columns([2, 5, 1])
        with col_label:
            st.markdown(label)
        with col_bar:
            st.progress(float(prob))
        with col_val:
            st.markdown(f"{prob:.1%}")

    st.caption(
        "Each bar shows the model's softmax probability for that class. The "
        "four probabilities sum to 100%. The final diagnosis is the "
        "highest-probability class (bolded above)."
    )


def render_visual_comparison(original_rgb, overlay_rgb):
    st.subheader("2. Visual Explanation: Original vs. Grad-CAM")

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            original_rgb,
            caption="Original X-ray (resized to model input, 224x224)",
            use_container_width=True,
        )
    with col2:
        st.image(
            overlay_rgb,
            caption="Grad-CAM Heatmap Overlay",
            use_container_width=True,
        )

    st.caption(
        "**How to read the heatmap:** warmer colors (red / yellow) mark "
        "regions of the X-ray that contributed most strongly to the "
        "model's prediction. Cooler colors (blue / dark) mark regions that "
        "had little to no influence."
    )


def render_clinical_notes(explanation):

    st.subheader("3. Clinical Transparency Notes")

    predicted_class = explanation["predicted_class"]
    affected_regions = explanation["affected_regions"]
    max_activation = explanation["max_activation"]
    class_probabilities = explanation["class_probabilities"]
    confidence = explanation["confidence"]

    # Plain-language, per-pathology context for the radiological pattern each
    # class is typically associated with. These describe *typical* imaging
    # findings and are explanatory only -- they are not diagnostic criteria.
    pathology_context = {
        "COVID19": (
            "COVID-19 pneumonia on chest X-ray is classically associated with "
            "bilateral, peripheral ground-glass opacities, often most "
            "prominent in the lower zones."
        ),
        "PNEUMONIA": (
            "Bacterial/viral pneumonia typically appears as localized areas "
            "of increased opacity (infiltrates or lobar consolidation) within "
            "the lung fields."
        ),
        "TURBERCULOSIS": (
            "Tuberculosis often favors the upper lobes and can present with "
            "cavitation, nodular opacities, or fibrotic changes."
        ),
    }

    region_text = ", ".join(affected_regions) if affected_regions else None

    if predicted_class == "NORMAL":
        st.markdown(
            "The model predicted **NORMAL** — the absence of a strong, "
            "localized activation pattern of the kind associated with the "
            "three disease classes (COVID-19, pneumonia, tuberculosis). The "
            "Grad-CAM overlay shows where the model's attention was "
            "concentrated, even though no disease pattern reached a "
            "deciding probability."
        )
        if region_text:
            st.caption(
                f"For reference, the model's strongest attention was in the "
                f"**{region_text}**."
            )
    else:
        context = pathology_context.get(predicted_class, "")
        if region_text:
            st.markdown(
                f"The model differentiated **{predicted_class}** from the "
                f"other three classes primarily on the basis of activity in "
                f"the **{region_text}**, shown as the warmest areas of the "
                f"Grad-CAM overlay above. {context}"
            )
        else:
            st.markdown(
                f"The model predicted **{predicted_class}** based on a more "
                f"diffusely distributed activation pattern across the image "
                f"rather than a single sharply localized region. {context}"
            )

    # Multi-pathology differentiation note: surface the runner-up class so the
    # reviewer can see how decisive (or borderline) the call was.
    ranked = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) >= 2:
        runner_up_name, runner_up_prob = ranked[1]
        margin = confidence - runner_up_prob
        if margin < 0.15:
            st.warning(
                f"**Differential note:** this was a relatively close call. The "
                f"next most likely class was **{runner_up_name}** "
                f"({runner_up_prob:.1%}), only {margin:.1%} behind the "
                f"predicted **{predicted_class}** ({confidence:.1%}). When two "
                f"classes are this close, the prediction should be treated as "
                f"low-confidence and warrants careful expert review."
            )
        else:
            st.info(
                f"**Differential note:** the model separated **{predicted_class}** "
                f"({confidence:.1%}) from the next most likely class "
                f"(**{runner_up_name}**, {runner_up_prob:.1%}) by a margin of "
                f"{margin:.1%}."
            )

    # st.metric("Grad-CAM localization strength", f"{max_activation:.1%}")
    # st.caption(
    #     "Localization strength is the peak normalized Grad-CAM activation "
    #     "found anywhere in the image, for the predicted class. Higher values "
    #     "indicate a sharply localized region of interest; lower values "
    #     "indicate the model's attention was spread more diffusely."
    # )

   


def main():
    render_header()
    render_sidebar()

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"No trained model found at `{MODEL_PATH}`. Train a model "
            f"first by running `python train.py` from the project root -- "
            f"this will save a model to that location."
        )
        st.stop()

    try:
        model, base_model = load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - surface any load error to the user
        st.error(f"Failed to load the model from `{MODEL_PATH}`: {exc}")
        st.stop()

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an image above to get started.")
        return

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        st.error(f"Could not read the uploaded file as an image: {exc}")
        st.stop()

    image_array = np.array(pil_image)

    with st.spinner("Analyzing X-ray..."):
        img_array, resized_rgb_uint8 = load_and_preprocess_single_image(image_array)
        explanation = generate_explanation(img_array, resized_rgb_uint8, model, base_model)

    st.markdown("---")
    render_prediction(explanation)

    st.markdown("---")
    render_visual_comparison(resized_rgb_uint8, explanation["overlay"])

    st.markdown("---")
    render_clinical_notes(explanation)


if __name__ == "__main__":
    main()
