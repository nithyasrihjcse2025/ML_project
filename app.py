import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("accident_severity_model.pkl")


model = load_model()


# =========================================================
# CITY → STATE MAPPING
# =========================================================

city_state = {
    "Chennai": "Tamil Nadu",
    "Mumbai": "Maharashtra",
    "Pune": "Maharashtra",
    "Chandigarh": "Punjab",
    "Delhi": "Delhi",
    "Bangalore": "Karnataka",
    "Hyderabad": "Telangana",
    "Kolkata": "West Bengal"
}


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 15px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .result-value {
        font-size: 28px;
        font-weight: 700;
    }

    .small-text {
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚗 Road Accident Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Accident Severity Prediction & Prevention System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("ℹ️ About the System")

    st.write(
        "This system uses Machine Learning to predict accident severity "
        "from road, traffic, weather and time-related conditions."
    )

    st.divider()

    st.subheader("🧠 ML Model")

    st.write("Random Forest Classifier")

    st.write("Prediction Classes:")

    st.write("• Minor")
    st.write("• Major")
    st.write("• Fatal")

    st.divider()

    st.subheader("🛡️ Prevention")

    st.write(
        "A separate rule-based safety module identifies risk factors "
        "and provides prevention recommendations."
    )


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📋 Road & Traffic Conditions</div>',
    unsafe_allow_html=True
)

st.write("Enter the conditions to generate a prediction.")

st.divider()


# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    city = st.selectbox(
        "🏙️ City",
        list(city_state.keys())
    )

    # Automatically assign state
    state = city_state[city]

    st.info(f"📍 State: **{state}**")

    road_type = st.selectbox(
        "🛣️ Road Type",
        ["highway", "urban", "rural"]
    )


with col2:

    hour = st.slider(
        "🕐 Hour of Day",
        0,
        23,
        12
    )

    day_of_week = st.selectbox(
        "📅 Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    lanes = st.slider(
        "↔️ Number of Lanes",
        1,
        6,
        4
    )


with col3:

    weather = st.selectbox(
        "🌦️ Weather",
        ["clear", "rain", "fog"]
    )

    visibility = st.selectbox(
        "👁️ Visibility",
        ["high", "medium", "low"]
    )

    traffic_density = st.selectbox(
        "🚦 Traffic Density",
        ["low", "medium", "high"]
    )


# =========================================================
# ROW 2
# =========================================================

st.divider()

col4, col5, col6 = st.columns(3)


with col4:

    cause = st.selectbox(
        "⚠️ Main Cause",
        [
            "weather",
            "distraction",
            "overspeeding",
            "drunk driving",
            "poor road"
        ]
    )


with col5:

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=27.0,
        step=0.5
    )


with col6:

    traffic_signal = st.selectbox(
        "🚥 Traffic Signal Available?",
        ["Yes", "No"]
    )


# =========================================================
# DERIVED FEATURES
# =========================================================

is_weekend = (
    1 if day_of_week in ["Saturday", "Sunday"]
    else 0
)

is_peak_hour = (
    1
    if (7 <= hour <= 9 or 17 <= hour <= 20)
    else 0
)

traffic_signal_value = (
    1 if traffic_signal == "Yes"
    else 0
)


# =========================================================
# CONDITION INDICATORS
# =========================================================

st.divider()

st.subheader("📌 Detected Conditions")

condition1, condition2, condition3, condition4 = st.columns(4)

with condition1:
    if is_weekend:
        st.success("📅 Weekend")
    else:
        st.info("📅 Weekday")

with condition2:
    if is_peak_hour:
        st.warning("🚦 Peak Hour")
    else:
        st.info("🕐 Non-Peak Hour")

with condition3:
    if weather == "rain":
        st.warning("🌧️ Rain")
    elif weather == "fog":
        st.warning("🌫️ Fog")
    else:
        st.success("☀️ Clear Weather")

with condition4:
    if visibility == "low":
        st.error("👁️ Low Visibility")
    else:
        st.success("👁️ Good Visibility")


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍  PREDICT ACCIDENT RISK",
    use_container_width=True
)


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk():

    score = 0

    factors = []

    recommendations = []

    # ---------------- WEATHER ----------------

    if weather == "rain":

        score += 2

        factors.append("🌧️ Rainy weather")

        recommendations.append(
            "Reduce speed and maintain extra distance on wet roads."
        )

    elif weather == "fog":

        score += 3

        factors.append("🌫️ Foggy weather")

        recommendations.append(
            "Use appropriate headlights and reduce speed in fog."
        )

    # ---------------- VISIBILITY ----------------

    if visibility == "low":

        score += 3

        factors.append("👁️ Low visibility")

        recommendations.append(
            "Drive slowly and maintain a greater following distance."
        )

    elif visibility == "medium":

        score += 1

        factors.append("👁️ Medium visibility")

    # ---------------- TRAFFIC ----------------

    if traffic_density == "high":

        score += 2

        factors.append("🚦 High traffic density")

        recommendations.append(
            "Avoid sudden lane changes in heavy traffic."
        )

    elif traffic_density == "medium":

        score += 1

        factors.append("🚦 Medium traffic density")

    # ---------------- TIME ----------------

    if hour >= 22 or hour <= 5:

        score += 2

        factors.append("🌙 Night-time driving")

        recommendations.append(
            "Use appropriate headlights and remain alert at night."
        )

    # ---------------- ROAD ----------------

    if road_type == "highway":

        score += 1

        factors.append("🛣️ High-speed highway")

        recommendations.append(
            "Maintain a safe speed and avoid sudden maneuvers on highways."
        )

    # ---------------- CAUSE ----------------

    if cause == "overspeeding":

        score += 3

        factors.append("⚡ Overspeeding")

        recommendations.append(
            "Follow the prescribed speed limit."
        )

    elif cause == "drunk driving":

        score += 3

        factors.append("🚫 Impaired driving risk")

        recommendations.append(
            "Never drive while impaired; use a safe alternative."
        )

    elif cause == "distraction":

        score += 2

        factors.append("📱 Driver distraction")

        recommendations.append(
            "Avoid mobile-phone use and other distractions while driving."
        )

    elif cause == "poor road":

        score += 2

        factors.append("🛤️ Poor road condition")

        recommendations.append(
            "Reduce speed and watch carefully for road hazards."
        )

    elif cause == "weather":

        score += 2

        factors.append("🌦️ Weather-related condition")

    # ---------------- PEAK HOUR ----------------

    if is_peak_hour:

        score += 1

        factors.append("🚦 Peak-hour traffic")

        recommendations.append(
            "Allow extra travel time and maintain safe spacing."
        )

    # ---------------- RISK LEVEL ----------------

    if score >= 7:

        risk_level = "HIGH"

    elif score >= 4:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return score, risk_level, factors, recommendations


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "city": city,

        "state": state,

        "hour": hour,

        "day_of_week": day_of_week,

        "is_weekend": is_weekend,

        "road_type": road_type,

        "lanes": lanes,

        "traffic_signal": traffic_signal_value,

        "weather": weather,

        "visibility": visibility,

        "temperature": temperature,

        "traffic_density": traffic_density,

        "cause": cause,

        "is_peak_hour": is_peak_hour

    }])


    # -----------------------------------------------------
    # ML PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = max(probabilities) * 100


    probability_df = pd.DataFrame({

        "Severity": [
            x.upper()
            for x in classes
        ],

        "Probability": probabilities * 100

    })


    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    score, risk_level, factors, recommendations = calculate_risk()


    # =====================================================
    # RESULTS HEADER
    # =====================================================

    st.divider()

    st.header("📊 Prediction Results")


    # =====================================================
    # RESULT CARDS
    # =====================================================

    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "🎯 Predicted Severity",
            prediction.upper()
        )


    with result2:

        st.metric(
            "📈 Model Confidence",
            f"{confidence:.2f}%"
        )


    with result3:

        st.metric(
            "⚠️ Risk Score",
            score
        )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    st.subheader("🚦 Overall Risk Level")


    if risk_level == "HIGH":

        st.error(
            "🔴 HIGH RISK — Immediate safety precautions recommended."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "🟠 MEDIUM RISK — Additional caution is recommended."
        )

    else:

        st.success(
            "🟢 LOW RISK — Continue following standard road-safety practices."
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader("📈 ML Severity Probabilities")

    chart_df = probability_df.set_index("Severity")

    st.bar_chart(chart_df)


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    st.subheader("📋 Prediction Summary")


    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.write(f"**🏙️ City:** {city}")

        st.write(f"**📍 State:** {state}")

        st.write(f"**🛣️ Road:** {road_type}")

        st.write(f"**↔️ Lanes:** {lanes}")


    with summary2:

        st.write(f"**🕐 Hour:** {hour}:00")

        st.write(f"**📅 Day:** {day_of_week}")

        st.write(f"**🌦️ Weather:** {weather}")

        st.write(f"**👁️ Visibility:** {visibility}")


    with summary3:

        st.write(f"**🚦 Traffic:** {traffic_density}")

        st.write(f"**⚠️ Cause:** {cause}")

        st.write(f"**🌡️ Temperature:** {temperature} °C")

        st.write(f"**🚥 Signal:** {traffic_signal}")


    # =====================================================
    # RISK FACTORS
    # =====================================================

    st.subheader("⚠️ Identified Risk Factors")


    if factors:

        for factor in factors:

            st.write(f"• {factor}")

    else:

        st.success(
            "No major rule-based risk factors identified."
        )


    # =====================================================
    # PREVENTION
    # =====================================================

    st.subheader("🛡️ Prevention Recommendations")


    if recommendations:

        unique_recommendations = list(
            dict.fromkeys(recommendations)
        )

        for recommendation in unique_recommendations:

            st.write(
                f"✅ {recommendation}"
            )

    else:

        st.success(
            "Maintain standard road-safety practices and follow traffic rules."
        )


    # =====================================================
    # TECHNICAL INFORMATION
    # =====================================================

    st.divider()

    with st.expander("🔬 Technical Information"):

        st.write(
            "**Machine Learning Model:** Random Forest Classifier"
        )

        st.write(
            "**Input Features:** Road, weather, traffic and time conditions"
        )

        st.write(
            "**Target:** Accident Severity"
        )

        st.write(
            "**Classes:** Minor, Major, Fatal"
        )

        st.write(
            "**Risk Module:** Rule-based safety assessment"
        )

        st.write(
            "The ML model predicts severity. The Risk Score and "
            "Prevention Recommendations are generated separately "
            "from predefined safety conditions."
        )
