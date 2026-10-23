
import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.mobilenet_finetuning import build_finetuning_model
from src.mobilenet_transfer import build_transfer_model


# ==========================================
# CONFIGURATION
# ==========================================

MODELS_DIR = Path("models")
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# raw_0_255:
# Images stay in range [0, 255].
# Use this if preprocessing is already inside the model,
# for example MobileNetV2 preprocess_input in a Lambda layer.
#
# divide_255:
# Images are divided by 255 before prediction.
# Use this if the model was trained with images normalized
# using image / 255.0 and has no internal preprocessing.

PREPROCESSING = {
    "fine_tuning_model.keras": "raw_0_255",
    "cnn_from_scratch_model.keras": "divide_255",
    "transfer_learning_model.keras": "raw_0_255",
}

# Assumption for a binary sigmoid model:
# output 0 = NORMAL, output 1 = PNEUMONIA
# This matches alphabetical folder ordering when using
# image_dataset_from_directory with NORMAL and PNEUMONIA.


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="ChestScan AI",
    page_icon="🩻",
    layout="wide",
)

st.title("ChestScan AI")
st.caption(
    "Chest X-ray classification — NORMAL vs PNEUMONIA"
)

st.warning(
    "Educational project only. Predictions are not a medical "
    "diagnosis and must not replace a qualified clinician."
)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_model(model_path):
    path = Path(model_path)
    model_builders = {
        "fine_tuning_model.keras": build_finetuning_model,
        "transfer_learning_model.keras": build_transfer_model,
    }

    builder = model_builders.get(path.name)
    if builder is not None:
        model = builder()
        model.load_weights(path)
        return model

    return tf.keras.models.load_model(
        path,
        compile=False,
        safe_mode=True,
    )


def discover_models():
    if not MODELS_DIR.exists():
        return {}

    files = sorted(MODELS_DIR.glob("*.keras"))

    return {
        file.stem: file
        for file in files
    }


models_paths = discover_models()

if not models_paths:
    st.error(
        "No .keras models found. Add your trained models "
        "to the 'models/' directory."
    )
    st.stop()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def prepare_image(image, model, preprocessing):
    image = image.convert("RGB")

    # Infer input dimensions from the model
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or 224
    width = input_shape[2] or 224

    image = image.resize((width, height))

    array = np.asarray(image, dtype=np.float32)

    if preprocessing == "divide_255":
        array = array / 255.0

    elif preprocessing == "raw_0_255":
        pass

    else:
        raise ValueError(
            f"Unknown preprocessing method: {preprocessing}"
        )

    return np.expand_dims(array, axis=0)


# ==========================================
# PREDICTION
# ==========================================

def predict_probabilities(model, image, preprocessing):
    x = prepare_image(image, model, preprocessing)

    output = np.asarray(model.predict(x, verbose=0))

    # Binary sigmoid output: shape (1, 1)
    if output.shape[-1] == 1:
        pneumonia_prob = float(output.reshape(-1)[0])

        probabilities = {
            "NORMAL": 1.0 - pneumonia_prob,
            "PNEUMONIA": pneumonia_prob,
        }

    # Two-class softmax output: shape (1, 2)
    elif output.shape[-1] == 2:
        values = output.reshape(-1)

        # Assumes class index 0 = NORMAL, index 1 = PNEUMONIA
        probabilities = {
            "NORMAL": float(values[0]),
            "PNEUMONIA": float(values[1]),
        }

    else:
        raise ValueError(
            "Unsupported output shape. Expected a binary "
            "sigmoid model or a two-class softmax model."
        )

    return probabilities


def get_preprocessing(model_path):
    return PREPROCESSING.get(
        model_path.name,
        "raw_0_255",
    )


# ==========================================
# MODEL SELECTION
# ==========================================

st.sidebar.header("Configuration")

model_names = list(models_paths.keys())

mode = st.sidebar.radio(
    "Prediction mode",
    [
        "Single model",
        "Global prediction (all models)",
    ],
)

if mode == "Single model":
    selected_models = [
        st.sidebar.selectbox(
            "Choose a model",
            model_names,
        )
    ]
else:
    selected_models = st.sidebar.multiselect(
        "Models to include in the ensemble",
        model_names,
        default=model_names,
    )

    if not selected_models:
        st.info("Select at least one model.")
        st.stop()


# ==========================================
# UPLOAD IMAGES
# ==========================================

st.subheader("Upload Chest X-ray images")

uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    run_prediction = st.button(
        "Run prediction",
        type="primary",
        use_container_width=True,
    )

    if run_prediction:
        results = []

        with st.spinner("Loading models and predicting..."):
            loaded_models = {}

            for name in selected_models:
                path = models_paths[name]
                loaded_models[name] = load_model(str(path))

            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file).convert("RGB")

                    model_results = []

                    for name, model in loaded_models.items():
                        preprocessing = get_preprocessing(
                            models_paths[name]
                        )

                        probs = predict_probabilities(
                            model,
                            image,
                            preprocessing,
                        )

                        predicted_class = max(
                            probs,
                            key=probs.get,
                        )

                        model_results.append({
                            "Image": uploaded_file.name,
                            "Model": name,
                            "Prediction": predicted_class,
                            "NORMAL (%)": round(
                                probs["NORMAL"] * 100, 2
                            ),
                            "PNEUMONIA (%)": round(
                                probs["PNEUMONIA"] * 100, 2
                            ),
                        })

                    # Average probability across selected models
                    avg_normal = np.mean([
                        row["NORMAL (%)"] / 100
                        for row in model_results
                    ])

                    avg_pneumonia = np.mean([
                        row["PNEUMONIA (%)"] / 100
                        for row in model_results
                    ])

                    final_class = (
                        "PNEUMONIA"
                        if avg_pneumonia >= avg_normal
                        else "NORMAL"
                    )

                    results.extend(model_results)

                    results.append({
                        "Image": uploaded_file.name,
                        "Model": "GLOBAL ENSEMBLE",
                        "Prediction": final_class,
                        "NORMAL (%)": round(
                            avg_normal * 100, 2
                        ),
                        "PNEUMONIA (%)": round(
                            avg_pneumonia * 100, 2
                        ),
                    })

                except Exception as error:
                    st.error(
                        f"Error processing {uploaded_file.name}: "
                        f"{error}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

        if results:
            df = pd.DataFrame(results)

            st.success("Prediction completed.")

            # ======================================
            # RESULTS PER IMAGE
            # ======================================

            for filename in df["Image"].unique():
                st.divider()
                st.subheader(f"Image: {filename}")

                current = df[df["Image"] == filename]

                final_row = current[
                    current["Model"] == "GLOBAL ENSEMBLE"
                ]

                if not final_row.empty:
                    row = final_row.iloc[0]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Final predicted class",
                            row["Prediction"],
                        )

                    with col2:
                        st.metric(
                            "NORMAL probability",
                            f'{row["NORMAL (%)"]:.2f}%',
                        )

                    with col3:
                        st.metric(
                            "PNEUMONIA probability",
                            f'{row["PNEUMONIA (%)"]:.2f}%',
                        )

                    st.image(
                        Image.open(
                            next(
                                file for file in uploaded_files
                                if file.name == filename
                            )
                        ).convert("RGB"),
                        caption=filename,
                        width=300,
                    )

                st.markdown("**Model comparison**")

                comparison = current[
                    current["Model"] != "GLOBAL ENSEMBLE"
                ].drop(columns=["Image"])

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                )

            # ======================================
            # GLOBAL TABLE
            # ======================================

            st.divider()
            st.subheader("All predictions")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="chestscan_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
