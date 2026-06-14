# 🫁 Explainable AI for Chest X-ray Diagnosis (4-Class)

An end-to-end, production-grade **Explainable AI (XAI)** system that classifies
chest X-rays into **four** classes — **COVID-19, Normal, Pneumonia, and
Tuberculosis** — using transfer learning, and — just as importantly —
*explains every prediction* with a Grad-CAM heatmap that highlights the lung
regions driving the model's decision for the predicted disease.


---

## Features

- **Transfer learning** with a MobileNetV2 backbone (ImageNet-pretrained),
  chosen for its small footprint (~3.4M params) and CPU-friendly inference.
- **4-class softmax head** distinguishing COVID-19, Normal, Pneumonia, and
  Tuberculosis, trained with categorical cross-entropy on one-hot labels.
- **Clinically-aware augmentation** — only anatomically plausible transforms
  (small rotations/zooms, horizontal flip, mild contrast jitter); no vertical
  flips, and black-fill borders that match an X-ray's dark background.
- **Class-imbalance handling** via automatically-computed per-class weights,
  since the dataset is heavily imbalanced (tuberculosis is the rarest class).
- **Disease-specific Grad-CAM** — the heatmap targets the gradient of the
  *winning* class, so it localizes the pathology for whichever disease was
  actually detected, and is robust to model save/reload cycles.
- **Clinical Transparency Notes** — plain-language explanations of *which*
  lung regions drove each prediction, *how* localized the attention was, and a
  multi-pathology differential note (the runner-up class and margin).
- **Full multi-class evaluation** — per-class / macro / weighted Precision,
  Recall, F1, One-vs-Rest ROC-AUC, a 4×4 confusion matrix, and per-class ROC
  curves, with a built-in explanation of why (macro) Recall is prioritized.
- **Streamlit clinical dashboard** — upload an X-ray, get a prediction, a
  full 4-class probability breakdown, and a side-by-side original-vs-heatmap.

---

## Architecture

```
            ┌─────────────────────────────────────────────────────┐
            │                  Input X-ray image                  │
            │              (any size, RGB or grayscale)           │
            └───────────────────────────┬─────────────────────────┘
                                        │
                        resize → 224×224, to 3-channel RGB
                                        │
                   ┌────────────────────┴────────────────────┐
                   │  Augmentation (TRAIN ONLY)              │
                   │  flip / rotate / zoom / contrast        │
                   └────────────────────┬────────────────────┘
                                        │
                       MobileNetV2 preprocess_input → [-1, 1]
                                        │
            ┌───────────────────────────┴─────────────────────────┐
            │   MobileNetV2 backbone (frozen, ImageNet weights)   │
            │                                                     │
            │   ... depthwise-separable conv blocks ...           │
            │                                                     │
            │   final activation "out_relu"  ──►  7 × 7 × 1280    │ ◄── Grad-CAM tap
            └───────────────────────────┬─────────────────────────┘
                                        │
                          GlobalAveragePooling2D        "global_avg_pool"
                                        │
                          Dense(128, relu)              "dense_head"
                                        │
                          BatchNormalization            "head_batchnorm"
                                        │
                          Dropout(0.3)                  "head_dropout"
                                        │
                          Dense(4, softmax)             "disease_probabilities"
                                        │
                   ┌────────────────────┴────────────────────┐
                   │  P(COVID19), P(NORMAL),                 │
                   │  P(PNEUMONIA), P(TURBERCULOSIS)         │  (sums to 1)
                   └─────────────────────────────────────────┘
```

Grad-CAM "taps" the `out_relu` feature map (7×7×1280) and weights it by the
gradient of the **winning class's** score w.r.t. that map — producing a 7×7
importance heatmap that is upsampled and overlaid on the original X-ray. A
different predicted disease yields a different heatmap.

---

## 📁 Project Structure

```
chest-xray-xai/
├── README.md                  # You are here
├── requirements.txt           # Version-locked dependencies
├── .gitignore
├── LICENSE                    # MIT (+ medical disclaimer)
├── train.py                   # Two-stage training entry point
├── evaluate.py                # Multi-class metrics, confusion matrix, ROC curves
├── app.py                     # Streamlit clinical dashboard
├── smoke_test.py              # Fast end-to-end wiring test (synthetic data)
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py       # tf.data pipelines, augmentation, class weights
│   ├── model.py               # MobileNetV2 + 4-way softmax head, fine-tuning
│   └── explainability.py      # Disease-specific Grad-CAM + region summaries
├── data/
│   └── chest_xray
├── models/                    # Saved .h5 models 
│   └── .gitkeep
└── outputs/                   # Training plots, logs, eval figures 
    └── .gitkeep
```

---

## Quick Start

### Clone and set up the environment

```bash
git clone <your-repo-url> chest-xray-xai
cd chest-xray-xai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Download the dataset

Follow the instructions in [`data/README.md`](data/README.md) to download the
Kaggle Chest X-Ray (Pneumonia, Covid-19, Tuberculosis) dataset into
`data/chest_xray/`.

---

## Usage

### Train

Stage 1 only (frozen backbone, head-only training — the default):

```bash
python train.py --data_dir data/chest_xray --epochs 15
```

Stage 2 fine-tuning (unfreeze top backbone layers at a lower learning rate):

```bash
python train.py --data_dir data/chest_xray --epochs 15 --fine_tune_epochs 10
```

### Launch the clinical dashboard

```bash
streamlit run app.py
```
---

## Explainability & Clinical Transparency

The XAI design rests on four ideas:

 **Grad-CAM at the right layer, for the right class.** Explanations are
   computed from the backbone's final convolutional feature map (`out_relu`,
   7×7×1280) — the last layer that still preserves spatial structure. Crucially,
   the gradient is taken with respect to the **winning class's** score
   (`argmax` of the 4-way softmax), so the heatmap localizes the detected
   disease specifically rather than some fixed class.


---

## Evaluation Metrics

`evaluate.py` reports the following on the held-out test set:


tes set performance metrics (multi class) 

Overall Accuracy:     -    0.8638

### 1. Per-Class Performance Summary

| Class | Precision | Recall (Sensitivity) | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **COVID-19** | 0.7123 | **0.9811** | 0.8254 | 106 |
| **NORMAL** | 0.9326 | 0.7094 | 0.8058 | 234 |
| **PNEUMONIA** | 0.8798 | 0.9385 | 0.9082 | 390 |
| **TUBERCULOSIS** | **0.9677** | 0.7317 | 0.8333 | 41 |

* **Macro-Recall Summary:** Achieved an overall **0.84 Macro-Recall** and **0.98 OvR ROC-AUC** across all categories.

---

### 2. Confusion Matrix Analytics
> **Rows = Actual Patient Condition** | **Columns = Model Predicted Diagnosis**

| Actual \ Predicted | COVID-19 | NORMAL | PNEUMONIA | TUBERCULOSIS |
| :--- | :---: | :---: | :---: | :---: |
| **COVID-19** | **104** | 0 | 1 | 1 |
| **NORMAL** | 19 | **166** | 49 | 0 |
| **PNEUMONIA** | 12 | 12 | **366** | 0 |
| **TUBERCULOSIS** | 11 | 0 | 0 | **30** |

### 3. Full Multi-Class Classification Report

| Diagnostic Category | Precision | Recall (Sensitivity) | F1-Score | Test Support (Images) |
| :--- | :---: | :---: | :---: | :---: |
| **COVID-19** | 0.71 | **0.98** | 0.83 | 106 |
| **NORMAL** | 0.93 | 0.71 | 0.81 | 234 |
| **PNEUMONIA** | 0.88 | **0.94** | 0.91 | 390 |
| **TUBERCULOSIS** | **0.97** | 0.73 | 0.83 | 41 |
| **Accuracy** | — | — | **0.86** | **771** |
| **Macro Average** | 0.87 | **0.84** | 0.84 | 771 |
| **Weighted Average** | 0.88 | 0.86 | 0.86 | 771 |

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model, training, Grad-CAM gradients
- **MobileNetV2** — pretrained backbone (transfer learning)
- **OpenCV** — heatmap resizing, colormapping, overlay compositing
- **scikit-learn** — multi-class evaluation metrics
- **Matplotlib / Seaborn** — training curves, confusion matrix, ROC curves
- **Streamlit** — clinical dashboard UI
- **NumPy / Pillow** — array and image handling

---

## 📜 License

Released under the [MIT License](LICENSE), with an additional notice that this
software is for educational and research use only and is not a medical device.
