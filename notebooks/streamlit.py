import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys


# Find the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Full model path
MODEL_PATH = BASE_DIR / "artifacts" / "flight_price_model.joblib"

st.write("Python:", sys.executable)
st.write("Model path:", str(MODEL_PATH))
st.write("File exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

st.success("Model loaded successfully!")


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Price Prediction")
st.write("Enter the flight information to predict the ticket price.")


# -------------------------
# Input variables
# -------------------------

airline = st.selectbox(
    "Airline",
    [
        "AirAsia",
        "Air_India",
        "GO_FIRST",
        "Indigo",
        "SpiceJet",
        "Vistara"
    ]
)

source_city = st.selectbox(
    "Source City",
    [
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai"
    ]
)

departure_time = st.selectbox(
    "Departure Time",
    [
        "Early_Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
        "Late_Night"
    ]
)

arrival_time = st.selectbox(
    "Arrival Time",
    [
        "Early_Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
        "Late_Night"
    ]
)

destination_city = st.selectbox(
    "Destination City",
    [
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai"
    ]
)

flight_class = st.selectbox(
    "Class",
    [
        "Economy",
        "Business"
    ]
)

stops = st.selectbox(
    "Stops",
    [
        "zero",
        "one",
        "two_or_more"
    ]
)

duration = st.number_input(
    "Duration (hours)",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1
)

days_left = st.number_input(
    "Days Before Departure",
    min_value=1,
    max_value=50,
    value=20,
    step=1
)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    # Create DataFrame with exactly
    # the same columns used during training
    input_data = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "departure_time": [departure_time],
        "arrival_time": [arrival_time],
        "destination_city": [destination_city],
        "class": [flight_class],
        "stops": [stops],
        "duration": [duration],
        "days_left": [days_left]
    })

    # Pipeline performs preprocessing automatically
    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(
        f"Predicted Flight Price: ₹{predicted_price:,.2f}"
    )