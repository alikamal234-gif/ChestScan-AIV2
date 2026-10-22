import tensorflow as tf
import kagglehub
import os

def load_data():
    path = kagglehub.dataset_download(
        "paultimothymooney/chest-xray-pneumonia"
    )

    data_dir = os.path.join(path, "chest_xray")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        validation_split=0.2,
        subset="training",
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=42
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        shuffle=False
    )

    return train_ds, val_ds, test_ds