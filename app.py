import streamlit as st
import xgboost as xgb
import numpy as np
import joblib

st.set_page_config(page_title="Uskuna Nosozligi Bashoratchisi", page_icon="⚙️", layout="wide")

# --- Orqa fon va umumiy stil ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        repeating-linear-gradient(45deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 15px),
        repeating-linear-gradient(-45deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 15px);
    pointer-events: none;
    z-index: 0;
}
h1, h2, h3, p, label, .stMarkdown {
    color: #f0f0f0 !important;
}
[data-testid="stMetricValue"] {
    color: #ffd166 !important;
}
.author-box {
    background: rgba(255,255,255,0.07);
    border-left: 3px solid #ffd166;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Uskuna Nosozligini Bashorat Qilish Tizimi")

# --- Sidebar: muallif ma'lumoti ---
with st.sidebar:
    st.markdown("""
    <div class="author-box">
    ⚙️ <b>Loyiha muallifi:</b><br>Ashirov Dostonbek<br><br>
    🎓 Energetika muhandisligi fakulteti<br>4-kurs<br><br>
    📧 pperceptron@gmail.com<br><br>
    🔗 <a href="https://github.com/doston-npp-ml-engineer" style="color:#ffd166;" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# --- Modellarni yuklaymiz (ilova ishga tushganda bir marta) ---
@st.cache_resource
def load_models():
    m1 = xgb.XGBClassifier()
    m1.load_model('failure_model.json')

    m2 = xgb.XGBClassifier()
    m2.load_model('reason_model.json')

    le = joblib.load('label_encoder.pkl')
    return m1, m2, le

model, model_reason, le = load_models()

st.subheader("Sensor ko'rsatkichlarini kiriting")

col1, col2 = st.columns(2)
with col1:
    type_ = st.selectbox("Mahsulot turi", ["L", "M", "H"])
    air_temp = st.number_input("Havo harorati (K)", value=298.0)
    process_temp = st.number_input("Jarayon harorati (K)", value=308.0)
with col2:
    rot_speed = st.number_input("Aylanish tezligi (rpm)", value=1500)
    torque = st.number_input("Moment (Nm)", value=40.0)
    tool_wear = st.number_input("Asbob eskirishi (min)", value=100)

type_map = {"L": 0, "M": 1, "H": 2}

if st.button("Bashorat qilish"):
    X_input = np.array([[type_map[type_], air_temp, process_temp,
                          rot_speed, torque, tool_wear]])

    proba_fail = model.predict_proba(X_input)[0][1]

    st.metric("Buzilish ehtimoli", f"{proba_fail*100:.1f}%")

    if proba_fail > 0.5:
        st.error("⚠️ Yuqori xavf! Mashina buzilishi mumkin.")

        reason_pred = model_reason.predict(X_input)[0]
        reason_name = le.inverse_transform([reason_pred])[0]

        reason_full = {
            "TWF": "Asbob eskirishi (Tool Wear Failure)",
            "HDF": "Issiqlik chiqmasligi (Heat Dissipation Failure)",
            "PWF": "Quvvat muammosi (Power Failure)",
            "OSF": "Haddan tashqari zo'riqish (Overstrain Failure)"
        }
        st.warning(f"Ehtimoliy sabab: **{reason_full.get(reason_name, reason_name)}**")
    else:
        st.success("✅ Mashina normal holatda ishlashi kutilmoqda.")
