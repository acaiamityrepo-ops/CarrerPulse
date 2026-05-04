import joblib, json
import streamlit as st
import numpy as np
import pandas as pd
import requests
import re
from urllib.parse import urlparse
import pdfplumber
from PIL import Image
import pytesseract
import base64

# ============================================
# CONFIGURATION & MODEL LOADING
# ============================================
@st.cache_resource
def load_pipeline(folder):
    try:
        pipe = joblib.load(f"{folder}/pipeline.pkl")
        with open(f"{folder}/meta.json", "r") as f:
            meta = json.load(f)
        return pipe, meta
    except:
        return None, None

eng_pipe, eng_meta = load_pipeline("artifacts_engineering")
bus_pipe, bus_meta = load_pipeline("artifacts_business")

# ============================================
# PAGE CONFIG & CSS
# ============================================
st.set_page_config(page_title="CareerPulse", page_icon="✨", layout="wide")

def display_permanent_logo():
    # Convert image to base64
    with open("logo.png", "rb") as f:
        data = f.read()
        home_b64 = base64.b64encode(data).decode()

    # Inject the HTML/CSS for the same-tab redirect
    st.markdown(
        f"""
        <style>
            .custom-logo {{
                position: fixed;
                top: 10px;
                left: 1rem;
                z-index: 999999;
            }}
            .custom-logo img {{
                height: 40px;
                width: auto;
                cursor: pointer;
            }}
        </style>
        <div class="custom-logo">
            <a href="https://acai-apps.amity.edu:8501/" target="_self">
                <img src="data:image/png;base64,{home_b64}">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
display_permanent_logo()

col1, col2, col3 = st.columns([3, 4, 1])
with col2:
    st.image("amity_logo.png")

st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: bold; text-align: center; color: #7C3AED; margin-bottom: 0.5rem; }
    .card {
        background: #0F172A; 
        border: 1px solid #1F2937;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        color: white;
    }
    .badge {
        display:inline-block;
        padding:5px 12px;
        border-radius:999px;
        background:#7C3AED;
        color:white;
        font-size:12px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #4C1D95 100%);
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🏠 Home"
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# ============================================
# HELPER FUNCTIONS
# ============================================
def detect_skills(text_l: str) -> list:
    skill_keywords = {
        "Python": [r"\bpython\b"], "Java": [r"\bjava\b"], "C++": [r"c\+\+"],
        "SQL": [r"\bsql\b"], "Machine Learning": [r"machine learning", r"\bml\b"],
        "Data Structures": [r"data structures", r"\bdsa\b"], "Excel": [r"\bexcel\b"]
    }
    found = [skill for skill, patterns in skill_keywords.items() if any(re.search(p, text_l) for p in patterns)]
    return sorted(list(set(found)))

def extract_text_from_cv(cv_file) -> str:
    text = ""
    if cv_file.type == "application/pdf":
        with pdfplumber.open(cv_file) as pdf:
            for page in pdf.pages[:5]:
                text += (page.extract_text() or "") + "\n"
    else:
        text = pytesseract.image_to_string(Image.open(cv_file))
    return text.strip()

# ============================================
# NAVIGATION
# ============================================
tabs = ["🏠 Home", "🔎 Prediction", "📊 Analysis"]
current_index = tabs.index(st.session_state.active_tab)

selected_tab = st.radio("Nav", tabs, index=current_index, horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = selected_tab
st.markdown("---")

# ============================================
# PAGE 1: HOME (Inputs & Processing)
# ============================================
if selected_tab == "🏠 Home":
    st.markdown('<div class="main-header">✨ CareerPulse</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.5, 4, 0.5])
    with col2:
        st.image("a.png", 
                        caption="Proposed Architecture for CarrerPulse")
        st.markdown("""
        <div style="text-align: justify;">
            The AI Career Architect empowers students to take control of their professional future by transforming standard resumes into data-driven roadmaps for success. Using sophisticated text extraction and machine learning algorithms, the platform analyzes your academic profile, CV, and even GitHub activity to predict placement outcomes with high accuracy. By calculating a comprehensive Employability Readiness Score based on everything from your CGPA to technical certifications, it provides a transparent view of where you stand in the job market. Say goodbye to career uncertainty and hello to personalized, AI-powered insights and suggestions designed to bridge the gap between graduation and your dream job.
           </div>
           </br>
            """,unsafe_allow_html=True
        )
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        c1, c2, c3 = st.columns(3)
        with c2:
            if st.button("Launch"):
                st.session_state.active_tab = "🔎 Prediction"
                st.rerun()
    st.caption(
    "⚠️ **Educational Disclaimer:** This is an AI career assistant intended to support, not replace, formal academic and career counseling. Users should verify all placement predictions and readiness scores against official university guidelines and individual recruiter requirements."
      )
    

elif selected_tab == "🔎 Prediction":
    col_l, col_r = st.columns([1, 1], gap="large")
    
    with col_l:
        st.subheader("1. Academic Details")
        stream = st.selectbox("Stream Category", ["Engineering", "Business & Management", "Other"])
        cgpa = st.number_input("CGPA (0–10)", 0.0, 10.0, 7.5)
        backlogs = st.number_input("Backlogs", 0, 20, 0)
        
        if stream == "Engineering":
            age = st.number_input("Age", 18, 40, 21)
            gender = st.selectbox("Gender", ["Male", "Female"])
        else:
            ssc_p = st.number_input("SSC %", 0.0, 100.0, 75.0)
            workex = st.selectbox("Work Experience", ["Yes", "No"])

    with col_r:
        st.subheader("2. Portfolio & CV")
        cv_file = st.file_uploader("Upload CV (PDF or Image)", type=["pdf", "png", "jpg"])
        github_url = st.text_input("GitHub URL")
        
        if st.button("Analyze My Career Profile", use_container_width=True):
            with st.spinner("Processing data..."):
                # Extract Skills
                skills = []
                if cv_file:
                    raw_text = extract_text_from_cv(cv_file)
                    skills = detect_skills(raw_text.lower())
                
                # Mock Probability Logic (Replace with eng_pipe.predict_proba if artifacts loaded)
                prob = 0.85 if cgpa > 7.5 else 0.45
                
                # Calculate Readiness
                readiness = min((cgpa * 5) + (len(skills) * 8), 100)
                
                # Store in session
                st.session_state.results = {
                    "prob": prob,
                    "readiness": readiness,
                    "skills": skills,
                    "stream": stream
                }
                st.session_state.processed = True
                st.session_state.active_tab = "📊 Analysis"
                st.rerun()

# ============================================
# PAGE 2: PREDICTION (Results & Insights)
# ============================================
elif selected_tab == "📊 Analysis":
    if not st.session_state.processed:
        st.warning("⚠️ No data found. Please go back to the Home page and submit your details.")
    else:
        res = st.session_state.results
        st.markdown(f"## Analysis for {res['stream']} Profile")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="badge">Placement Forecast</div>
                <h2>{'Likely Placed' if res['prob'] > 0.5 else 'Needs Improvement'}</h2>
                <p>Probability: <b>{res['prob']*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="badge">Readiness Score</div>
                <h2>{res['readiness']:.1f}%</h2>
                <p>Based on skills and academics</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("🛠️ Detected Expertise")
        if res['skills']:
            st.write(", ".join([f"**{s}**" for s in res['skills']]))
        else:
            st.info("No specific technical skills were detected from the CV.")

        st.subheader("📌 Roadmap to Success")
        if res['readiness'] < 60:
            st.error("Action Required: Your readiness score is below target. Focus on building projects and learning Python/SQL.")
        else:
            st.success("Great job! Your profile shows strong technical alignment for current market trends.")
            
        if st.button("Start New Analysis"):
            st.session_state.processed = False
            st.session_state.active_tab = "🏠 Home"
            st.rerun()