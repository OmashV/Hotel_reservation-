import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Hotel Reservation Prediction", layout="centered")
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2em;
    }
    .stNumberInput, .stSelectbox, .stCheckbox {
        max-width: 300px;
        margin: auto;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-will-cancel {
        background-color: #FFCDD2;
        color: #D32F2F;
        border: 2px solid #D32F2F;
    }
    .prediction-will-not-cancel {
        background-color: #C8E6C9;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
    .prediction-text {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .probability-text {
        font-size: 1.2em;
        color: #37474F;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏨 Hotel Reservation Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a reservation will be canceled with ease</div>', unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        label_encoders = joblib.load("models/label_encoders.pkl")
        poly = joblib.load("models/poly.pkl")
        return model, scaler, label_encoders, poly
    except FileNotFoundError:
        st.error("Model or preprocessing files not found. Please ensure 'best_model.pkl', 'scaler.pkl', 'label_encoders.pkl', and 'poly.pkl' are in the 'models' directory.")
        st.stop()

model, scaler, label_encoders, poly = load_artifacts()

# Define feature options based on the dataset
categorical_options = {
    'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
    'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'],
    'market_segment_type': ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'],
    'lead_time_category': ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
}

# Define all possible input features
all_input_features = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
    'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
    'lead_time', 'arrival_month', 'market_segment_type', 'repeated_guest',
    'avg_price_per_room', 'no_of_special_requests'
]

# Get expected features from scaler
try:
    expected_features = scaler.feature_names_in_
except AttributeError:
    expected_features = all_input_features + [
        'total_guests', 'total_nights', 'price_per_night', 
        'is_weekend_booking', 'lead_time_category', 
        'loyalty_score', 'has_special_requests', 
        'lead_time_price_interaction', 'market_segment_special_requests',
        'lead_time avg_price_per_room'
    ]

# Create input form
st.header("Enter Reservation Details")
with st.form("reservation_form"):
    col1, col2 = st.columns([1, 1])

    input_data = {}
    with col1:
        if 'no_of_adults' in all_input_features:
            input_data['no_of_adults'] = st.number_input("Number of Adults", min_value=0, max_value=10, value=2)
        if 'no_of_children' in all_input_features:
            input_data['no_of_children'] = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        if 'no_of_weekend_nights' in all_input_features:
            input_data['no_of_weekend_nights'] = st.number_input("Weekend Nights", min_value=0, max_value=7, value=0)
        if 'no_of_week_nights' in all_input_features:
            input_data['no_of_week_nights'] = st.number_input("Week Nights", min_value=0, max_value=17, value=2)
        if 'type_of_meal_plan' in all_input_features:
            input_data['type_of_meal_plan'] = st.selectbox("Meal Plan", categorical_options['type_of_meal_plan'])
        if 'room_type_reserved' in all_input_features:
            input_data['room_type_reserved'] = st.selectbox("Room Type", categorical_options['room_type_reserved'])
        if 'arrival_month' in all_input_features:
            input_data['arrival_month'] = st.number_input("Arrival Month", min_value=1, max_value=12, value=1)

    with col2:
        if 'market_segment_type' in all_input_features:
            input_data['market_segment_type'] = st.selectbox("Market Segment", categorical_options['market_segment_type'])
        if 'lead_time' in all_input_features:
            input_data['lead_time'] = st.number_input("Lead Time (days)", min_value=0, max_value=1000, value=30)
        if 'avg_price_per_room' in all_input_features:
            input_data['avg_price_per_room'] = st.number_input("Avg Price per Room", min_value=0.0, max_value=1000.0, value=100.0)
        if 'no_of_special_requests' in all_input_features:
            input_data['no_of_special_requests'] = st.number_input("Special Requests", min_value=0, max_value=10, value=0)
        if 'repeated_guest' in all_input_features:
            input_data['repeated_guest'] = int(st.checkbox("Repeated Guest"))
        if 'required_car_parking_space' in all_input_features:
            input_data['required_car_parking_space'] = int(st.checkbox("Requires Parking"))

    submit_button = st.form_submit_button("Predict", use_container_width=True)

# Preprocessing function
def preprocess_input(data, label_encoders, scaler, poly):
    # Create DataFrame
    df = pd.DataFrame([data])

    # Feature engineering
    if 'no_of_adults' in df.columns and 'no_of_children' in df.columns:
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    if 'no_of_weekend_nights' in df.columns and 'no_of_week_nights' in df.columns:
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    if 'avg_price_per_room' in df.columns and 'total_nights' in df.columns:
        df['price_per_night'] = df['avg_price_per_room'] / (df['total_nights'] + 1)
    if 'no_of_weekend_nights' in df.columns:
        df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)
    if 'lead_time' in df.columns:
        df['lead_time_category'] = pd.cut(df['lead_time'], 
                                         bins=[0, 7, 30, 90, 365, 1000],
                                         labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long'])
    if 'repeated_guest' in df.columns:
        df['loyalty_score'] = df['repeated_guest']
    if 'no_of_special_requests' in df.columns:
        df['has_special_requests'] = (df['no_of_special_requests'] > 0).astype(int)
    if 'lead_time' in df.columns and 'avg_price_per_room' in df.columns:
        df['lead_time_price_interaction'] = df['lead_time'] * df['avg_price_per_room']

    # Encode categorical variables
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'lead_time_category']
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders.get(col)
            if le:
                most_frequent = le.classes_[0]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else most_frequent)
                df[col] = le.transform(df[col].astype(str))

    # Create interaction term for market_segment_type * no_of_special_requests
    if 'market_segment_type' in df.columns and 'no_of_special_requests' in df.columns:
        df['market_segment_special_requests'] = df['no_of_special_requests']
        if 'market_segment_type' in label_encoders:
            le = label_encoders['market_segment_type']
            most_frequent = le.classes_[0]
            df['market_segment_type_encoded'] = df['market_segment_type'].apply(lambda x: x if x in le.classes_ else most_frequent)
            df['market_segment_type_encoded'] = le.transform(df['market_segment_type_encoded'].astype(str))
            df['market_segment_special_requests'] = df['market_segment_type_encoded'] * df['no_of_special_requests']
            df = df.drop(columns=['market_segment_type_encoded'])

    # Add polynomial features
    poly_features = ['lead_time', 'avg_price_per_room']
    if all(f in df.columns for f in poly_features):
        poly_cols = [f for f in poly_features if f in df.columns]
        poly_df = pd.DataFrame(
            poly.transform(df[poly_cols]),
            columns=poly.get_feature_names_out(poly_cols),
            index=df.index
        )
        df = df.drop(columns=poly_cols)
        df = pd.concat([df, poly_df], axis=1)

    # Select features in the correct order
    feature_order = [f for f in expected_features if f in df.columns]
    df = df[feature_order]

    # Verify feature order matches scaler's expectations
    if list(df.columns) != list(expected_features):
        st.error(f"Feature mismatch! Expected features: {expected_features}, Got: {list(df.columns)}")
        st.stop()

    # Scale numerical features
    df_scaled = scaler.transform(df)
    df_scaled = np.nan_to_num(df_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return df_scaled

# Prediction
if submit_button:
    try:
        # Preprocess input
        processed_data = preprocess_input(input_data, label_encoders, scaler, poly)
        
        # Predict probability
        prob = model.predict_proba(processed_data)[:, 1][0]
        
        # Apply custom threshold
        threshold = 0.2  # Lowered to boost recall
        prediction = 1 if prob >= threshold else 0
        
        # Display result
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-box prediction-will-cancel">'
                f'<div class="prediction-text">Prediction: Will Cancel</div>'
                f'<div class="probability-text">Probability of Cancellation: {prob*100:.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box prediction-will-not-cancel">'
                f'<div class="prediction-text">Prediction: Will Not Cancel</div>'
                f'<div class="probability-text">Probability of Cancellation: {prob*100:.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")