# Dataset Setup

This project uses the **Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)**
dataset on Kaggle:

> https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis

It is a **4-class** dataset (~7,135 images, ~1.93 GB). The code in
`src/data_pipeline.py` expects the data at `data/chest_xray/` (relative to the
project root) with the following layout:

```
data/chest_xray/
├── train/
│   ├── COVID19/          (*.jpeg / *.jpg / *.png)
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
├── val/
│   ├── COVID19/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
└── test/
    ├── COVID19/
    ├── NORMAL/
    ├── PNEUMONIA/
    └── TURBERCULOSIS/
```

> **⚠️ Spelling matters:** the tuberculosis folder is literally named
> **`TURBERCULOSIS`** (with an extra "R") in this dataset. `CLASS_NAMES` in
> `src/data_pipeline.py` matches that exact spelling on purpose, because the
> `class_names` passed to `image_dataset_from_directory` must match the
> on-disk folder names character-for-character. If you rename the folder, you
> must update `CLASS_NAMES` to match (and vice versa).

## Option A — Kaggle CLI (recommended)

1. Install and authenticate the Kaggle CLI:

   ```bash
   pip install kaggle
   # Place your kaggle.json API token at ~/.kaggle/kaggle.json
   # (Account -> Settings -> API -> "Create New Token" on kaggle.com)
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. From the **project root**, download and unzip into `data/`:

   ```bash
   kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis -p data/
   unzip -q data/chest-xray-pneumoniacovid19tuberculosis.zip -d data/
   ```

3. Confirm (or arrange) that the extracted `train/`, `val/`, and `test/`
   folders sit directly inside `data/chest_xray/` as shown above. Depending on
   the archive's internal structure you may need to move the extracted folders
   into a `data/chest_xray/` directory so the paths match what
   `src/data_pipeline.py` expects (or pass a different `--data_dir`).

## Option B — Manual download

Download the ZIP from the Kaggle page above, then extract it so that the
`train/`, `val/`, and `test/` directories sit inside `data/chest_xray/`
(matching the tree above).

## A note on class imbalance

The four classes are substantially imbalanced (for example, PNEUMONIA is the
largest class while TURBERCULOSIS is by far the smallest). This is handled
automatically by `compute_class_weights()` in `src/data_pipeline.py`, which
`train.py` passes to `model.fit(..., class_weight=...)`. The rarest class
(tuberculosis) is exactly the one a screening tool can least afford to miss,
which is why per-class weighting and **macro-averaged** Recall (see
`evaluate.py`) matter so much here — no manual rebalancing of the files is
required.

## A note on the validation split

If a particular split is small or noisy, validation-driven callbacks
(`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, all of which monitor
validation metrics) can behave erratically. If you find the `val/` split is
too small for stable model selection, consider pooling all images per class
and re-splitting into a stratified 80/10/10 (train/val/test) using
`sklearn.model_selection.train_test_split` with `stratify=labels`, preserving
the per-class ratios. The pipeline runs with the dataset as-shipped; this is
purely to make validation metrics more trustworthy.
