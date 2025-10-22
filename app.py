import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Heart Disease Predictor ❤️",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================
# Load Model & Scaler
# ==========================
model = joblib.load("heart_disease_model_basic.pkl")
scaler = joblib.load("scaler_basic.pkl")

# ==========================
# Header
# ==========================
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color:#e63946;'>❤️ Heart Disease Risk Predictor</h1>
        <p style='font-size:18px;'>Quickly assess heart disease risk with basic health details.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# Input Section (Two Columns)
# ==========================
st.subheader("🩺 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧓 Age", min_value=1, max_value=120, value=45)
    sex = st.radio("⚧ Sex", ["Female", "Male"], horizontal=True)
    sex_val = 0 if sex == "Female" else 1
    cp = st.selectbox("💢 Chest Pain Type", ["None", "Atypical", "Typical"])
    cp_val = ["None", "Atypical", "Typical"].index(cp)

with col2:
    trestbps = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("🧪 Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("❤️ Max Heart Rate Achieved", min_value=60, max_value=250, value=150)

st.markdown("---")

# ==========================
# Predict Button
# ==========================
if st.button("🔍 Predict Heart Disease Risk"):
    with st.spinner("Analyzing patient data..."):
        # Prepare input
        input_data = np.array([[age, sex_val, cp_val, trestbps, chol, thalach]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

    st.markdown("---")

    # ==========================
    # Display Result
    # ==========================
    if prediction[0] == 1:
        st.error("⚠️ **High Risk:** The patient may have heart disease.\n\n👉 Immediate medical consultation is recommended.")
        st.markdown(
            """
            <div style='background-color:#ffe5e5;color:black; padding:15px; border-radius:10px;'>
                <b>Tips:</b><br>
                • Reduce salt & saturated fats 🍟<br>
                • Engage in daily exercise 🏃‍♂️<br>
                • Get regular checkups 🏥
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **Low Risk:** The patient is unlikely to have heart disease.")
        st.markdown(
            """
            <div style='background-color:#e0f7e0; color:black; padding:15px; border-radius:10px;'>
                <b>Health Tips:</b><br>
                • Maintain a balanced diet 🥗<br>
                • Keep stress low 🧘‍♀️<br>
                • Continue regular health monitoring ❤️
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px;'>
        Built with 💻 using <b>Streamlit</b> & <b>Machine Learning</b><br>
        Stay Heart Healthy ❤️
    </div>
    """,
    unsafe_allow_html=True
)
