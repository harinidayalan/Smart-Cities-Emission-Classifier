
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------
# 1. Page configuration
# ---------------------------------

st.set_page_config(
    page_title="Smart Cities Prediction",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Smart Cities - Industrial Emission Support Prediction")
st.write(
    "Enter your survey responses to predict the support category "
    "using the trained machine learning model."
)

# ---------------------------------
# 2. File paths
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "X_features_raw.csv"

# ---------------------------------
# 3. Load model and reference data
# ---------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)

try:
    if not MODEL_PATH.exists():
        st.error("Model file not found: " + str(MODEL_PATH))
        st.stop()

    if not DATA_PATH.exists():
        st.error("Dataset file not found: " + str(DATA_PATH))
        st.stop()

    model = load_model()
    reference_df = load_reference_data()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------------------------
# 4. Create input form
# ---------------------------------

st.subheader("Enter Survey Details")

input_values = {}

with st.form("prediction_form"):

    for column in reference_df.columns:

        st.markdown(f"**{column}**")

        values = (
            reference_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            input_values[column] = st.selectbox(
                label=column,
                options=values,
                key=f"input_{column}"
            )
        else:
            input_values[column] = st.text_input(
                label=column,
                key=f"input_{column}"
            )

    submitted = st.form_submit_button(
        "Predict Support Category"
    )

# ---------------------------------
# 5. Prediction
# ---------------------------------

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=reference_df.columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if str(prediction) == "1":
            st.success("Prediction: Support")
        elif str(prediction) == "0":
            st.warning("Prediction: Not Support")
        else:
            st.info(f"Predicted category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Prediction Confidence")

            for label, probability in zip(
                model.classes_, probabilities
            ):
                st.write(
                    f"Class {label}: {probability * 100:.2f}%"
                )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------------------------
# 6. Project information
# ---------------------------------

st.divider()

st.subheader("About This Project")

st.write(
    "This application uses survey responses and a trained "
    "machine learning model to predict public support categories "
    "related to industrial emission monitoring."
)

st.caption(
    "This is a survey-based prediction system, not an actual "
    "industrial smoke detection system."
)
