
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_transfer_model(input_shape=(224, 224, 3)):

    # 1. Charger le modèle pré-entraîné
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # 2. Geler les poids pré-entraînés
    base_model.trainable = False

    # 3. Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ], name="data_augmentation")

    # 4. Construire l'architecture avec Sequential
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        data_augmentation,

        tf.keras.layers.Lambda(preprocess_input),

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # 5. Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model
