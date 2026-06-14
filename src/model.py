import tensorflow as tf

from src.data_pipeline import IMAGE_SIZE, NUM_CLASSES

#: The 3-channel RGB input shape expected by the model, derived from
#: ``IMAGE_SIZE`` in ``src.data_pipeline`` so the two modules can never fall
#: out of sync.
IMAGE_SHAPE = (*IMAGE_SIZE, 3)


def build_model(
    input_shape=IMAGE_SHAPE,
    dropout_rate=0.3,
    dense_units=128,
    learning_rate=1e-4,
    freeze_backbone=True,
    weights="imagenet",
):
    """Builds and compiles the MobileNetV2-based pneumonia classifier.

    Architecture::

        Input(input_shape)
          -> MobileNetV2(include_top=False, weights=weights)       [base_model]
          -> GlobalAveragePooling2D                                 "global_avg_pool"
          -> Dense(dense_units, activation="relu")                  "dense_head"
          -> BatchNormalization                                     "head_batchnorm"
          -> Dropout(dropout_rate)                                  "head_dropout"
          -> Dense(NUM_CLASSES, activation="softmax")               "disease_probabilities"


    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=input_shape, name="xray_input")

  
    x = base_model(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batchnorm")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="disease_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="chest_xray_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )

    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=30, learning_rate=1e-5):
 
    base_model.trainable = True

    # Freeze every layer except the last `num_layers`.
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # Within the newly-unfrozen portion, keep BatchNormalization layers
    # frozen so their pretrained running statistics are preserved.
    for layer in base_model.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )

    return model
