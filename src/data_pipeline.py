import glob
import os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Global constants


#: Target (height, width) that every image is resized to before being fed to
#: MobileNetV2. MobileNetV2 was pretrained at 224x224, and using the same
#: resolution maximizes the value of the transferred ImageNet features.
IMAGE_SIZE = (224, 224)

#: Default batch size used for training, validation, and test datasets.
BATCH_SIZE = 32

SEED = 42


CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]


NUM_CLASSES = len(CLASS_NAMES)


def _build_augmentation_pipeline():
   
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02, fill_mode="constant"),
            tf.keras.layers.RandomZoom(
                height_factor=0.1, width_factor=0.1, fill_mode="constant"
            ),
            tf.keras.layers.RandomContrast(factor=0.1),
        ],
        name="clinical_augmentation",
    )


def _prepare(dataset, augment, augmentation_layer):
 
    if augment:
        dataset = dataset.map(
            lambda images, labels: (augmentation_layer(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.map(
        lambda images, labels: (preprocess_input(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def get_datasets(data_dir="data/chest_xray", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED):
   
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    augmentation_layer = _build_augmentation_pipeline()

    train_ds = _prepare(train_ds, augment=True, augmentation_layer=augmentation_layer)
    val_ds = _prepare(val_ds, augment=False, augmentation_layer=None)
    test_ds = _prepare(test_ds, augment=False, augmentation_layer=None)

    return train_ds, val_ds, test_ds, CLASS_NAMES


def _count_images(class_dir):

    count = 0
    for pattern in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
        count += len(glob.glob(os.path.join(class_dir, pattern)))
    return count


def compute_class_weights(data_dir="data/chest_xray"):

    train_dir = os.path.join(data_dir, "train")

    counts = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(train_dir, class_name)
        counts[class_idx] = _count_images(class_dir)

    for class_idx, count in counts.items():
        if count == 0:
            raise ValueError(
                f"No images found for class '{CLASS_NAMES[class_idx]}' in "
                f"'{os.path.join(train_dir, CLASS_NAMES[class_idx])}'. "
                f"Please verify that the dataset has been downloaded and "
                f"extracted correctly (see data/README.md)."
            )

    total = sum(counts.values())
    n_classes = len(CLASS_NAMES)

    class_weights = {
        class_idx: total / (n_classes * count) for class_idx, count in counts.items()
    }

    return class_weights


def load_and_preprocess_single_image(image, image_size=IMAGE_SIZE):
   
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Normalize to 3-channel RGB regardless of the input's channel layout.
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    resized = tf.image.resize(image, image_size, method="bilinear")
    resized = tf.clip_by_value(resized, 0.0, 255.0)

    resized_rgb_uint8 = tf.cast(resized, tf.uint8).numpy()

    preprocessed = preprocess_input(resized)
    preprocessed_array = tf.expand_dims(preprocessed, axis=0).numpy()

    return preprocessed_array, resized_rgb_uint8
