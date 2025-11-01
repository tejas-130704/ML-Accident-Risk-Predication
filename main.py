import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import QuantileTransformer

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="Accident Risk Prediction", page_icon="⚙️", layout="wide")

st.title("⚙️ Interactive Accident Risk Predictor")
st.write("Adjust the sliders and dropdowns below — predictions update live with a 95% confidence interval.")

# ------------------------- MODEL LOADING -------------------------
@st.cache_resource
def load_models():
    model_lower = joblib.load("model_lower.pkl")
    model_upper = joblib.load("model_upper.pkl")
    model = joblib.load("model.pkl")
    return model_lower, model_upper, model

model_lower, model_upper, model = load_models()
qt = joblib.load("quantile_transformer.pkl")

# ------------------------- LAYOUT: INPUT (LEFT) & OUTPUT (RIGHT) -------------------------
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("🧾 Input Road & Environment Details")

    col1, col2 = st.columns(2)
    with col1:
        num_lanes = st.slider("Number of Lanes", 1, 8, 2)
        curvature = st.slider("Road Curvature", 0.0, 1.0, 0.3)
        speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 60)
        lighting = st.selectbox("Lighting Condition", ["daylight", "dim", "night"])
        weather = st.selectbox("Weather Condition", ["clear", "rainy", "foggy"])
        road_signs_present = st.selectbox("Road Signs Present", [True, False])

    with col2:
        public_road = st.selectbox("Is it a Public Road?", [True, False])
        time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening"])
        holiday = st.selectbox("Is it a Holiday?", [True, False])
        school_season = st.selectbox("School Season Active?", [True, False])
        num_reported_accidents = st.number_input("Reported Accidents (past)", 0, 100, 5)
        road_type = st.selectbox("Road Type", ["rural", "urban"])

# ------------------------- FEATURE ENGINEERING -------------------------
data = pd.DataFrame([{
    "road_type": road_type,
    "num_lanes": num_lanes,
    "curvature": curvature,
    "speed_limit": speed_limit,
    "lighting": lighting,
    "weather": weather,
    "road_signs_present": road_signs_present,
    "public_road": public_road,
    "time_of_day": time_of_day,
    "holiday": holiday,
    "school_season": school_season,
    "num_reported_accidents": num_reported_accidents
}])

data["holiday"] = data["holiday"].map({False: 0, True: 1})
data["road_signs_present"] = data["road_signs_present"].map({False: 0, True: 1})
data["public_road"] = data["public_road"].map({False: 0, True: 1})
data["time_of_day"] = data["time_of_day"].map({"morning": 1, "afternoon": 2, "evening": 3})
data["lighting"] = data["lighting"].replace({"daylight": 0, "dim": 1, "night": 2})
data["is_night"] = data["lighting"].replace({"daylight": 0, "dim": 0, "night": 1})
data["is_blurly_wheather"] = data["weather"].isin(["rainy", "foggy"]).astype(int)
data["high_curvature"] = data["curvature"].apply(lambda x: 1 if x > 0.5 else 0)
data["is_high_speed_limit"] = data["speed_limit"].apply(lambda x: 1 if x >= 60 else 0)
data["new_normalize_curvature"] = qt.transform(np.array(data["curvature"]).reshape(-1, 1))
data["road_lane_ratio"] = data["num_lanes"] / (data["speed_limit"] + 1)
data["high_risk_score"] = (
    data["is_blurly_wheather"] +
    data["is_night"] +
    data["is_high_speed_limit"] +
    data["high_curvature"]
).astype(int)
data["is_public_and_blurly"] = (data["is_blurly_wheather"] & data["public_road"]).astype(int)
data["night_and_holiday"] = (data["is_night"] & data["holiday"]).astype(int)
data["road_type_rural"] = (data["road_type"].str.lower() == "rural").astype(int)
data["road_type_urban"] = (data["road_type"].str.lower() == "urban").astype(int)
data["weather_foggy"] = (data["weather"] == "foggy").astype(int)
data["weather_rainy"] = (data["weather"] == "rainy").astype(int)
data["risk_factor_index"] = data["high_risk_score"] * data["road_lane_ratio"]

final_features = [
    'num_lanes', 'curvature', 'speed_limit', 'lighting',
    'road_signs_present', 'public_road', 'time_of_day', 'holiday',
    'school_season', 'num_reported_accidents', 'road_type_rural',
    'road_type_urban', 'new_normalize_curvature', 'high_curvature',
    'is_high_speed_limit', 'is_night', 'is_blurly_wheather',
    'weather_foggy', 'weather_rainy', 'road_lane_ratio',
    'is_public_and_blurly', 'high_risk_score', 'night_and_holiday',
    'risk_factor_index'
]

X = data[final_features].copy()
for col in model_lower.feature_name_:
    if col not in X.columns:
        X[col] = 0
X = X[model_lower.feature_name_]

# ------------------------- LIVE PREDICTION -------------------------
with right_col:
    st.subheader("📊 Live Prediction Results")
    try:
        y_lower = model_lower.predict(X)[0]
        y_upper = model_upper.predict(X)[0]
        y_mean = (y_lower + y_upper) / 2

        st.success("✅ Prediction Updated Live")
        st.metric("Predicted Risk (Mean)", f"{y_mean:.4f}")
        st.info(f"**95% Confidence Interval:** [{y_lower:.3f}, {y_upper:.3f}]")

        # Dynamic risk level
        if y_mean < 0.3:
            risk_label = "🟢 Low Risk"
        elif y_mean < 0.7:
            risk_label = "🟡 Moderate Risk"
        else:
            risk_label = "🔴 High Risk"
        st.markdown(f"### {risk_label}")

        # Confidence chart
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Accident Risk"], [y_upper - y_lower], left=y_lower, color="orange", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted Risk Range (95% CI)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

st.markdown("---")
