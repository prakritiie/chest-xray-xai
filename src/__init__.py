"""
Explainable AI for Chest X-ray Multi-Disease Diagnosis
======================================================

This package contains the core modules for the chest X-ray multi-class
classification project (4 classes: COVID19, NORMAL, PNEUMONIA,
TURBERCULOSIS):

- ``data_pipeline``: dataset loading, augmentation, preprocessing, and
  class-weight computation.
- ``model``: MobileNetV2-based transfer-learning model with a 4-way softmax
  classification head.
- ``explainability``: disease-specific Grad-CAM heatmap generation and
  clinical-transparency utilities.
"""
