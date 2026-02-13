import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle, os, json

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "color_theme" not in st.session_state:
    st.session_state.color_theme = "Indigo"
if "language" not in st.session_state:
    st.session_state.language = "English"

# ─────────────────────────────────────────────────────────
# COLOR THEMES
# ─────────────────────────────────────────────────────────
COLOR_THEMES = {
    "Indigo":  {"primary": "#4f46e5", "secondary": "#7c3aed", "accent": "#ec4899", "chart1": "#4f46e5", "chart2": "#ec4899", "chart3": "#10b981"},
    "Ocean":   {"primary": "#0891b2", "secondary": "#0e7490", "accent": "#06b6d4", "chart1": "#0891b2", "chart2": "#06b6d4", "chart3": "#f59e0b"},
    "Emerald": {"primary": "#059669", "secondary": "#047857", "accent": "#10b981", "chart1": "#059669", "chart2": "#34d399", "chart3": "#8b5cf6"},
    "Rose":    {"primary": "#e11d48", "secondary": "#be123c", "accent": "#fb7185", "chart1": "#e11d48", "chart2": "#fb7185", "chart3": "#6366f1"},
    "Amber":   {"primary": "#d97706", "secondary": "#b45309", "accent": "#f59e0b", "chart1": "#d97706", "chart2": "#fbbf24", "chart3": "#8b5cf6"},
}

# ─────────────────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "app_title": "🫀 PulseGuard AI",
        "hero_desc": "Intelligent Blood Pressure Prediction System — harnessing the power of machine learning to predict and classify hypertension stages, enabling early risk identification and preventive healthcare decisions through data-driven analysis.",
        "patient_records": "Patient Records",
        "best_accuracy": "Best Model Accuracy",
        "models_trained": "ML Models Trained",
        "hyp_stages": "Hypertension Stages",
        "feat_class_title": "Hypertension Classification",
        "feat_class_desc": "Classifies patients into Normal, Stage-1, Stage-2, and Hypertensive Crisis using ensemble ML models.",
        "feat_risk_title": "Risk Prediction",
        "feat_risk_desc": "Provides confidence-scored risk assessments based on 13 clinical and lifestyle parameters.",
        "feat_life_title": "Lifestyle Factor Analysis",
        "feat_life_desc": "Analyzes impact of diet, medication adherence, and activity levels on blood pressure.",
        "feat_ai_title": "AI Health Insights",
        "feat_ai_desc": "Generates personalized health recommendations powered by Groq LLM technology.",
        "feat_eda_title": "Interactive EDA",
        "feat_eda_desc": "Explore data patterns with interactive Plotly charts — distributions, correlations, and trends.",
        "feat_model_title": "Model Comparison",
        "feat_model_desc": "Compare Logistic Regression, Random Forest, and XGBoost performance side-by-side.",
        "use_cases": "Use Cases",
        "uc_screening": "Preventive Screening",
        "uc_screening_desc": "Early risk identification for healthy individuals",
        "uc_clinical": "Clinical Decision Support",
        "uc_clinical_desc": "Assists healthcare workers with data-driven insights",
        "uc_research": "Health Research",
        "uc_research_desc": "Pattern analysis for public health studies",
        "nav_home": "🏠 Home", "nav_eda": "📊 EDA Dashboard", "nav_model": "🤖 Model Performance",
        "nav_predict": "🩺 Predict Hypertension", "nav_about": "ℹ️ About",
        "eda_title": "📊 Exploratory Data Analysis",
        "eda_desc": "Interactive visualizations revealing patterns and insights in the hypertension dataset.",
        "model_title": "🤖 Model Performance",
        "model_desc": "Comparative analysis of three machine learning algorithms trained on the hypertension dataset.",
        "predict_title": "🩺 Predict Hypertension Stage",
        "predict_desc": "Enter patient information below to get AI-powered hypertension risk assessment.",
        "about_title": "ℹ️ About PulseGuard AI",
        "about_desc": "An intelligent blood pressure prediction system for early hypertension risk identification.",
        "project_overview": "🎯 Project Overview",
        "overview_text": "PulseGuard AI is an advanced machine learning project designed to predict and classify various stages of hypertension in patients based on clinical parameters and lifestyle factors. The system utilizes supervised learning algorithms to analyze patient health data and provide accurate predictions along with personalized medical recommendations powered by Groq's LLM technology.",
        "tech_used": "🛠️ Technologies Used",
        "future_impl": "🚀 Future Implementations",
        "disclaimer": "⚠️ DISCLAIMER: PulseGuard AI is designed for educational and screening purposes only. It is not a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare providers for medical decisions.",
        "predict_btn": "🔍 Predict Hypertension Stage",
        "risk_complete": "Risk Assessment Complete",
        "classification": "Classification",
        "confidence": "Confidence",
        "model_label": "Model",
        "prob_dist": "📊 Probability Distribution Across Stages",
        "ai_insights": "🤖 AI-Powered Health Insights",
        "ai_generating": "Generating personalized health recommendations with Groq AI...",
        "ai_assessment": "🧠 Personalized Health Assessment",
        "accuracy_comp": "📊 Accuracy Comparison",
        "perf_summary": "📋 Performance Summary",
        "detailed_reports": "🔍 Detailed Classification Reports",
        "conf_matrix": "Confusion Matrix",
        "model_rationale": "🏆 Model Selection Rationale",
        "gender_dist": "1️⃣ Gender Distribution",
        "stages_dist": "2️⃣ Hypertension Stages Distribution",
        "bp_corr": "3️⃣ Systolic vs Diastolic Correlation",
        "med_severity": "4️⃣ Medication Status vs Symptom Severity",
        "age_stages": "5️⃣ Age Group vs Hypertension Stages",
        "scatter_bp": "6️⃣ Systolic vs Diastolic by Hypertension Stage",
        "dark_mode": "🌙 Dark Mode",
        "language": "🌐 Language",
        "color_theme": "🎨 Theme",
    },
    "Hindi": {
        "app_title": "🫀 पल्सगार्ड AI",
        "hero_desc": "बुद्धिमान रक्तचाप भविष्यवाणी प्रणाली — उच्च रक्तचाप के चरणों की पहचान और वर्गीकरण के लिए मशीन लर्निंग का उपयोग।",
        "patient_records": "रोगी रिकॉर्ड",
        "best_accuracy": "सर्वश्रेष्ठ मॉडल सटीकता",
        "models_trained": "ML मॉडल प्रशिक्षित",
        "hyp_stages": "उच्च रक्तचाप चरण",
        "feat_class_title": "उच्च रक्तचाप वर्गीकरण",
        "feat_class_desc": "एंसेंबल ML मॉडल का उपयोग करके रोगियों को सामान्य, चरण-1, चरण-2, और संकट में वर्गीकृत करता है।",
        "feat_risk_title": "जोखिम भविष्यवाणी",
        "feat_risk_desc": "13 नैदानिक पैरामीटरों के आधार पर जोखिम मूल्यांकन प्रदान करता है।",
        "feat_life_title": "जीवनशैली विश्लेषण",
        "feat_life_desc": "आहार, दवा पालन और गतिविधि स्तरों के प्रभाव का विश्लेषण करता है।",
        "feat_ai_title": "AI स्वास्थ्य अंतर्दृष्टि",
        "feat_ai_desc": "Groq LLM तकनीक द्वारा संचालित व्यक्तिगत स्वास्थ्य सिफारिशें।",
        "feat_eda_title": "इंटरैक्टिव EDA",
        "feat_eda_desc": "इंटरैक्टिव Plotly चार्ट के साथ डेटा पैटर्न खोजें।",
        "feat_model_title": "मॉडल तुलना",
        "feat_model_desc": "तीन ML एल्गोरिदम की साथ-साथ तुलना करें।",
        "use_cases": "उपयोग के मामले",
        "uc_screening": "निवारक जांच",
        "uc_screening_desc": "स्वस्थ व्यक्तियों के लिए प्रारंभिक जोखिम पहचान",
        "uc_clinical": "नैदानिक निर्णय सहायता",
        "uc_clinical_desc": "डेटा-संचालित अंतर्दृष्टि के साथ स्वास्थ्य कार्यकर्ताओं की सहायता",
        "uc_research": "स्वास्थ्य अनुसंधान",
        "uc_research_desc": "सार्वजनिक स्वास्थ्य अध्ययन के लिए पैटर्न विश्लेषण",
        "nav_home": "🏠 होम", "nav_eda": "📊 EDA डैशबोर्ड", "nav_model": "🤖 मॉडल प्रदर्शन",
        "nav_predict": "🩺 भविष्यवाणी", "nav_about": "ℹ️ के बारे में",
        "eda_title": "📊 खोजपूर्ण डेटा विश्लेषण",
        "eda_desc": "उच्च रक्तचाप डेटासेट में पैटर्न प्रकट करने वाले इंटरैक्टिव विज़ुअलाइज़ेशन।",
        "model_title": "🤖 मॉडल प्रदर्शन",
        "model_desc": "डेटासेट पर प्रशिक्षित तीन एल्गोरिदम का तुलनात्मक विश्लेषण।",
        "predict_title": "🩺 उच्च रक्तचाप चरण की भविष्यवाणी",
        "predict_desc": "AI-संचालित जोखिम मूल्यांकन प्राप्त करने के लिए रोगी की जानकारी दर्ज करें।",
        "about_title": "ℹ️ PulseGuard AI के बारे में",
        "about_desc": "प्रारंभिक उच्च रक्तचाप जोखिम पहचान के लिए बुद्धिमान प्रणाली।",
        "project_overview": "🎯 परियोजना अवलोकन",
        "overview_text": "PulseGuard AI एक उन्नत मशीन लर्निंग परियोजना है जो नैदानिक मापदंडों और जीवनशैली कारकों के आधार पर उच्च रक्तचाप के विभिन्न चरणों की भविष्यवाणी और वर्गीकरण करती है।",
        "tech_used": "🛠️ उपयोग की गई तकनीकें",
        "future_impl": "🚀 भविष्य की योजनाएँ",
        "disclaimer": "⚠️ अस्वीकरण: PulseGuard AI केवल शैक्षिक और स्क्रीनिंग उद्देश्यों के लिए है। यह पेशेवर चिकित्सा निदान का विकल्प नहीं है।",
        "predict_btn": "🔍 उच्च रक्तचाप की भविष्यवाणी करें",
        "risk_complete": "जोखिम मूल्यांकन पूर्ण",
        "classification": "वर्गीकरण",
        "confidence": "विश्वास",
        "model_label": "मॉडल",
        "prob_dist": "📊 चरणों में संभावना वितरण",
        "ai_insights": "🤖 AI-संचालित स्वास्थ्य अंतर्दृष्टि",
        "ai_generating": "Groq AI के साथ व्यक्तिगत स्वास्थ्य सिफारिशें बना रहा है...",
        "ai_assessment": "🧠 व्यक्तिगत स्वास्थ्य मूल्यांकन",
        "accuracy_comp": "📊 सटीकता तुलना",
        "perf_summary": "📋 प्रदर्शन सारांश",
        "detailed_reports": "🔍 विस्तृत वर्गीकरण रिपोर्ट",
        "conf_matrix": "कन्फ्यूजन मैट्रिक्स",
        "model_rationale": "🏆 मॉडल चयन तर्क",
        "gender_dist": "1️⃣ लिंग वितरण",
        "stages_dist": "2️⃣ उच्च रक्तचाप चरण वितरण",
        "bp_corr": "3️⃣ सिस्टोलिक vs डायस्टोलिक सहसंबंध",
        "med_severity": "4️⃣ दवा स्थिति vs लक्षण गंभीरता",
        "age_stages": "5️⃣ आयु समूह vs उच्च रक्तचाप चरण",
        "scatter_bp": "6️⃣ चरण अनुसार सिस्टोलिक vs डायस्टोलिक",
        "dark_mode": "🌙 डार्क मोड",
        "language": "🌐 भाषा",
        "color_theme": "🎨 थीम",
    },
    "Telugu": {
        "app_title": "🫀 పల్స్‌గార్డ్ AI",
        "hero_desc": "తెలివైన రక్తపోటు అంచనా వ్యవస్థ — హైపర్‌టెన్షన్ దశలను అంచనా వేయడానికి మెషిన్ లెర్నింగ్ ఉపయోగించడం.",
        "patient_records": "రోగి రికార్డులు",
        "best_accuracy": "ఉత్తమ మోడల్ ఖచ్చితత్వం",
        "models_trained": "ML మోడల్స్ శిక్షణ",
        "hyp_stages": "హైపర్‌టెన్షన్ దశలు",
        "feat_class_title": "హైపర్‌టెన్షన్ వర్గీకరణ",
        "feat_class_desc": "ఎన్‌సెంబుల్ ML మోడల్‌లను ఉపయోగించి రోగులను సాధారణ, దశ-1, దశ-2 మరియు సంక్షోభంగా వర్గీకరిస్తుంది.",
        "feat_risk_title": "ప్రమాద అంచనా",
        "feat_risk_desc": "13 క్లినికల్ పారామీటర్ల ఆధారంగా ప్రమాద మూల్యాంకనాలు అందిస్తుంది.",
        "feat_life_title": "జీవనశైలి విశ్లేషణ",
        "feat_life_desc": "ఆహారం, మందుల వాడకం మరియు కార్యాచరణ ప్రభావాన్ని విశ్లేషిస్తుంది.",
        "feat_ai_title": "AI ఆరోగ్య అంతర్దృష్టులు",
        "feat_ai_desc": "Groq LLM సాంకేతిక పరిజ్ఞానంతో వ్యక్తిగత ఆరోగ్య సిఫార్సులు.",
        "feat_eda_title": "ఇంటరాక్టివ్ EDA",
        "feat_eda_desc": "ఇంటరాక్టివ్ Plotly చార్ట్‌లతో డేటా నమూనాలను అన్వేషించండి.",
        "feat_model_title": "మోడల్ పోలిక",
        "feat_model_desc": "మూడు ML అల్గారిథమ్‌ల పనితీరును పక్కపక్కన పోల్చండి.",
        "use_cases": "ఉపయోగ సందర్భాలు",
        "uc_screening": "నివారణ పరీక్ష",
        "uc_screening_desc": "ఆరోగ్యవంతులకు ముందస్తు ప్రమాద గుర్తింపు",
        "uc_clinical": "క్లినికల్ నిర్ణయ సహాయం",
        "uc_clinical_desc": "డేటా-ఆధారిత అంతర్దృష్టులతో వైద్య సిబ్బందికి సహాయం",
        "uc_research": "ఆరోగ్య పరిశోధన",
        "uc_research_desc": "ప్రజారోగ్య అధ్యయనాల కోసం నమూనా విశ్లేషణ",
        "nav_home": "🏠 హోమ్", "nav_eda": "📊 EDA డాష్‌బోర్డ్", "nav_model": "🤖 మోడల్ పనితీరు",
        "nav_predict": "🩺 అంచనా", "nav_about": "ℹ️ గురించి",
        "eda_title": "📊 అన్వేషణ డేటా విశ్లేషణ",
        "eda_desc": "హైపర్‌టెన్షన్ డేటాసెట్‌లో నమూనాలను వెల్లడించే ఇంటరాక్టివ్ విజువలైజేషన్‌లు.",
        "model_title": "🤖 మోడల్ పనితీరు",
        "model_desc": "డేటాసెట్‌పై శిక్షణ పొందిన మూడు అల్గారిథమ్‌ల తులనాత్మక విశ్లేషణ.",
        "predict_title": "🩺 హైపర్‌టెన్షన్ దశ అంచనా",
        "predict_desc": "AI-ఆధారిత ప్రమాద మూల్యాంకనం పొందడానికి రోగి సమాచారాన్ని నమోదు చేయండి.",
        "about_title": "ℹ️ PulseGuard AI గురించి",
        "about_desc": "ముందస్తు హైపర్‌టెన్షన్ ప్రమాద గుర్తింపు కోసం తెలివైన వ్యవస్థ.",
        "project_overview": "🎯 ప్రాజెక్ట్ అవలోకనం",
        "overview_text": "PulseGuard AI అనేది క్లినికల్ పారామీటర్లు మరియు జీవనశైలి కారకాల ఆధారంగా హైపర్‌టెన్షన్ దశలను అంచనా వేయడానికి మరియు వర్గీకరించడానికి రూపొందించబడిన అధునాతన మెషిన్ లెర్నింగ్ ప్రాజెక్ట్.",
        "tech_used": "🛠️ ఉపయోగించిన సాంకేతికతలు",
        "future_impl": "🚀 భవిష్యత్ అమలులు",
        "disclaimer": "⚠️ నిరాకరణ: PulseGuard AI విద్యా మరియు స్క్రీనింగ్ ప్రయోజనాల కోసం మాత్రమే. ఇది వృత్తిపరమైన వైద్య నిర్ధారణకు ప్రత్యామ్నాయం కాదు.",
        "predict_btn": "🔍 హైపర్‌టెన్షన్ అంచనా వేయండి",
        "risk_complete": "ప్రమాద మూల్యాంకనం పూర్తయింది",
        "classification": "వర్గీకరణ",
        "confidence": "విశ్వాసం",
        "model_label": "మోడల్",
        "prob_dist": "📊 దశల్లో సంభావ్యత పంపిణీ",
        "ai_insights": "🤖 AI-ఆధారిత ఆరోగ్య అంతర్దృష్టులు",
        "ai_generating": "Groq AI తో వ్యక్తిగత ఆరోగ్య సిఫార్సులు రూపొందిస్తోంది...",
        "ai_assessment": "🧠 వ్యక్తిగత ఆరోగ్య మూల్యాంకనం",
        "accuracy_comp": "📊 ఖచ్చితత్వ పోలిక",
        "perf_summary": "📋 పనితీరు సారాంశం",
        "detailed_reports": "🔍 వివరమైన వర్గీకరణ నివేదికలు",
        "conf_matrix": "కన్ఫ్యూజన్ మ్యాట్రిక్స్",
        "model_rationale": "🏆 మోడల్ ఎంపిక హేతువు",
        "gender_dist": "1️⃣ లింగ పంపిణీ",
        "stages_dist": "2️⃣ హైపర్‌టెన్షన్ దశల పంపిణీ",
        "bp_corr": "3️⃣ సిస్టోలిక్ vs డయాస్టోలిక్ సహసంబంధం",
        "med_severity": "4️⃣ మందుల స్థితి vs లక్షణ తీవ్రత",
        "age_stages": "5️⃣ వయసు సమూహం vs హైపర్‌టెన్షన్ దశలు",
        "scatter_bp": "6️⃣ దశ వారీగా సిస్టోలిక్ vs డయాస్టోలిక్",
        "dark_mode": "🌙 డార్క్ మోడ్",
        "language": "🌐 భాష",
        "color_theme": "🎨 థీమ్",
    },
}

T = TRANSLATIONS[st.session_state.language]
theme = COLOR_THEMES[st.session_state.color_theme]
dark = st.session_state.dark_mode

# ─────────────────────────────────────────────────────────
# TOP TOOLBAR
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,rgba(79,70,229,0.08),rgba(236,72,153,0.08));
    border-radius:12px;padding:0.3rem 1rem 0.1rem;margin-bottom:1rem;
    border:1px solid rgba(79,70,229,0.15);">
</div>
""", unsafe_allow_html=True)
tb1, tb2, tb3 = st.columns([1, 1, 1])
with tb1:
    st.markdown("**🌐 Language**")
    lang = st.selectbox("lang_sel", ["English", "Hindi", "Telugu"], index=["English", "Hindi", "Telugu"].index(st.session_state.language), label_visibility="collapsed", key="lang_select")
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.rerun()
with tb2:
    st.markdown("**🎨 Color Theme**")
    ct = st.selectbox("theme_sel", list(COLOR_THEMES.keys()), index=list(COLOR_THEMES.keys()).index(st.session_state.color_theme), label_visibility="collapsed", key="theme_select")
    if ct != st.session_state.color_theme:
        st.session_state.color_theme = ct
        st.rerun()
with tb3:
    st.markdown("**🌙 Dark Mode**")
    dm = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        st.rerun()
st.markdown("---")

# ─────────────────────────────────────────────────────────
# DYNAMIC CSS
# ─────────────────────────────────────────────────────────
if dark:
    bg_main = "#0f172a"
    bg_card = "#1e293b"
    bg_card_border = "#334155"
    text_main = "#f1f5f9"
    text_secondary = "#94a3b8"
    text_heading = "#f8fafc"
    shadow_color = "rgba(0,0,0,0.4)"
    chart_template = "plotly_dark"
    chart_bg = "rgba(0,0,0,0)"
    mpl_face = "#0f172a"
    mpl_text = "#f1f5f9"
    mpl_tick = "#94a3b8"
    table_border = "#334155"
else:
    bg_main = "#f8f9fc"
    bg_card = "#ffffff"
    bg_card_border = "#e5e7eb"
    text_main = "#1e293b"
    text_secondary = "#6b7280"
    text_heading = "#1e293b"
    shadow_color = "rgba(0,0,0,0.06)"
    chart_template = "plotly_white"
    chart_bg = ""
    mpl_face = "#ffffff"
    mpl_text = "#1e293b"
    mpl_tick = "#374151"
    table_border = "#e5e7eb"

pri = theme["primary"]
sec = theme["secondary"]
acc = theme["accent"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 0.5rem; }}

/* Force theme */
.stApp {{ background-color: {bg_main}; color: {text_main}; }}
h1, h2, h3, h4, h5, h6 {{ color: {text_heading} !important; }}
p, li, span, label, div {{ color: {text_main}; }}
.stMarkdown, .stMarkdown p, .stMarkdown li {{ color: {text_main} !important; }}
.stTabs [data-baseweb="tab"] {{ color: {text_main} !important; }}
.stTabs [aria-selected="true"] {{ color: {pri} !important; }}
.stMetric label {{ color: {text_secondary} !important; }}
.stMetric [data-testid="stMetricValue"] {{ color: {text_main} !important; }}
.stSelectbox label, .stRadio label {{ color: {text_secondary} !important; }}

/* Sidebar override for white text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] .stRadio label {{ color: #ffffff !important; }}
.hero-card, .hero-card h1, .hero-card p, .hero-card span {{ color: #ffffff !important; }}
.risk-normal, .risk-normal *, .risk-stage2, .risk-stage2 *, .risk-crisis, .risk-crisis * {{ color: #ffffff !important; }}
.risk-stage1, .risk-stage1 * {{ color: #1e293b !important; }}

/* Toolbar row */
.toolbar-row {{ display:flex; gap:1rem; align-items:center; padding:0.5rem 0; }}

/* Hero gradient card */
.hero-card {{
    background: linear-gradient(135deg, {pri} 0%, {sec} 40%, {acc} 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px {pri}4d;
    position: relative;
    overflow: hidden;
}}
.hero-card::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero-card h1 {{ font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem; }}
.hero-card p  {{ font-size: 1.15rem; opacity: 0.9; line-height: 1.6; max-width: 700px; }}

/* Metric cards */
.metric-card {{
    background: {bg_card};
    border: 1px solid {bg_card_border};
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px {shadow_color};
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 30px {pri}26; }}
.metric-card .value {{ font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, {pri}, {acc}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.metric-card .label {{ font-size: 0.85rem; color: {text_secondary}; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }}

/* Risk result cards */
.risk-normal  {{ background: linear-gradient(135deg, #10b981, #34d399); border-radius: 16px; padding: 2rem; color: white; }}
.risk-stage1  {{ background: linear-gradient(135deg, #f59e0b, #fbbf24); border-radius: 16px; padding: 2rem; color: #1e293b; }}
.risk-stage2  {{ background: linear-gradient(135deg, #ef4444, #f87171); border-radius: 16px; padding: 2rem; color: white; }}
.risk-crisis  {{ background: linear-gradient(135deg, #991b1b, #dc2626); border-radius: 16px; padding: 2rem; color: white; }}
.risk-title   {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; }}
.risk-stage   {{ font-size: 1.3rem; font-weight: 600; }}

/* Feature cards */
.feature-card {{
    background: {bg_card};
    border: 1px solid {bg_card_border};
    border-radius: 16px;
    padding: 1.8rem;
    height: 100%;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.feature-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px {pri}1f; }}
.feature-card .icon {{ font-size: 2.5rem; margin-bottom: 0.8rem; }}
.feature-card h3 {{ color: {pri} !important; font-size: 1.1rem; margin-bottom: 0.5rem; }}
.feature-card p  {{ color: {text_secondary} !important; font-size: 0.9rem; line-height: 1.5; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {pri}, {sec}, {pri}dd);
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.2); }}

/* Model comparison table */
.model-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 16px {shadow_color};
}}
.model-table th {{
    background: linear-gradient(135deg, {pri}, {sec});
    color: #ffffff;
    padding: 1rem;
    font-weight: 600;
    text-align: center;
}}
.model-table td {{
    padding: 0.8rem 1rem;
    text-align: center;
    border-bottom: 1px solid {table_border};
    background: {bg_card};
    color: {text_main};
}}

/* AI insight card */
.ai-insight {{
    background: {bg_card};
    border: 2px solid {pri}33;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 4px 20px {pri}14;
}}
.ai-insight h3 {{ background: linear-gradient(135deg, {pri}, {acc}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

/* About section */
.about-card {{
    background: {bg_card};
    border: 1px solid {bg_card_border};
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px {shadow_color};
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# GROQ CONFIG
# ─────────────────────────────────────────────────────────
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def get_groq_insight(prediction, confidence, patient_info):
    """Generate personalized health insight using Groq API."""
    if not GROQ_API_KEY:
        return "⚠️ **Groq API key not configured.** Set the `GROQ_API_KEY` environment variable to enable AI insights."
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are a medical AI assistant for hypertension risk assessment. 
A patient has been classified as: {prediction} with {confidence:.1f}% confidence.

Patient Profile:
- Gender: {patient_info.get('Gender', 'N/A')}
- Age Group: {patient_info.get('Age', 'N/A')}
- Family History: {patient_info.get('History', 'N/A')}
- Currently a Patient: {patient_info.get('Patient', 'N/A')}
- Takes Medication: {patient_info.get('TakeMedication', 'N/A')}
- Symptom Severity: {patient_info.get('Severity', 'N/A')}
- Shortness of Breath: {patient_info.get('BreathShortness', 'N/A')}
- Visual Changes: {patient_info.get('VisualChanges', 'N/A')}
- Nose Bleeding: {patient_info.get('NoseBleeding', 'N/A')}
- Controlled Diet: {patient_info.get('ControlledDiet', 'N/A')}
- Systolic Range: {patient_info.get('Systolic', 'N/A')}
- Diastolic Range: {patient_info.get('Diastolic', 'N/A')}

Provide a detailed, personalized health assessment including:
1. A brief explanation of their hypertension stage
2. Key risk factors identified from their profile
3. 5 specific lifestyle recommendations
4. When to seek immediate medical attention
5. Dietary suggestions specific to their condition

Keep the tone professional yet caring. Use bullet points and clear formatting. 
IMPORTANT: Add a disclaimer that this is AI-generated and not a substitute for professional medical advice."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not generate AI insight: {str(e)}"


# ─────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """Load dataset, clean, encode, and return processed data."""
    DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
    df = pd.read_csv(DATA_PATH)

    # Rename column
    df.rename(columns={"C": "Gender"}, inplace=True)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Fix inconsistencies
    df["Severity"] = df["Severity"].replace({"Sever": "Severe"})
    df["Systolic"] = df["Systolic"].replace({"121- 130": "121 - 130"})
    df["Stages"] = df["Stages"].replace({
        "HYPERTENSION (Stage-2).": "HYPERTENSION (Stage-2)",
        "HYPERTENSIVE CRISI": "HYPERTENSIVE CRISIS",
    })

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


@st.cache_data
def encode_data(df):
    """Encode categorical features and scale."""
    df_enc = df.copy()

    # Label mappings
    mappings = {
        "Gender":          {"Male": 0, "Female": 1},
        "Age":             {"18-34": 1, "35-50": 2, "51-64": 3, "65+": 4},
        "History":         {"No": 0, "Yes": 1},
        "Patient":         {"No": 0, "Yes": 1},
        "TakeMedication":  {"No": 0, "Yes": 1},
        "Severity":        {"Mild": 0, "Moderate": 1, "Severe": 2},
        "BreathShortness": {"No": 0, "Yes": 1},
        "VisualChanges":   {"No": 0, "Yes": 1},
        "NoseBleeding":    {"No": 0, "Yes": 1},
        "Whendiagnoused":  {"<1 Year": 0, "1 - 5 Years": 1, ">5 Years": 2},
        "Systolic":        {"100+": 0, "111 - 120": 1, "121 - 130": 2, "130+": 3},
        "Diastolic":       {"70 - 80": 0, "81 - 90": 1, "91 - 100": 2, "100+": 3, "130+": 4},
        "ControlledDiet":  {"No": 0, "Yes": 1},
        "Stages":          {"NORMAL": 0, "HYPERTENSION (Stage-1)": 1, "HYPERTENSION (Stage-2)": 2, "HYPERTENSIVE CRISIS": 3},
    }
    for col, mapping in mappings.items():
        df_enc[col] = df_enc[col].map(mapping)

    # Feature scaling (MinMax on ordinal features)
    features = df_enc.drop(columns=["Stages"])
    target = df_enc["Stages"]
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return features_scaled, target, scaler, mappings


# ─────────────────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_models(_X_train, _y_train, _X_test, _y_test):
    """Train all three models and return results."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        y_pred = model.predict(_X_test)
        acc = accuracy_score(_y_test, y_pred)
        report = classification_report(_y_test, y_pred, output_dict=True)
        cm = confusion_matrix(_y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc, "report": report, "confusion_matrix": cm, "predictions": y_pred}
    return results


# ─────────────────────────────────────────────────────────
# LOAD DATA & TRAIN
# ─────────────────────────────────────────────────────────
df_raw = load_and_clean_data()
X, y, scaler, label_mappings = encode_data(df_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_results = train_models(X_train, y_train, X_test, y_test)

# Reverse mapping for display
stage_labels = {0: "NORMAL", 1: "HYPERTENSION (Stage-1)", 2: "HYPERTENSION (Stage-2)", 3: "HYPERTENSIVE CRISIS"}

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 PulseGuard AI")
    st.markdown("---")
    nav_options = [T["nav_home"], T["nav_eda"], T["nav_model"], T["nav_predict"], T["nav_about"]]
    nav_keys = ["home", "eda", "model", "predict", "about"]
    page_sel = st.radio("Navigate", nav_options, label_visibility="collapsed")
    page_key = nav_keys[nav_options.index(page_sel)]
    st.markdown("---")
    st.markdown(
        '<p style="color:rgba(255,255,255,0.7);font-size:0.75rem;text-align:center;">© 2026 PulseGuard AI<br>Powered by Machine Learning</p>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════
if page_key == "home":
    st.markdown(f"""
    <div class="hero-card">
        <h1>{T['app_title']}</h1>
        <p>{T['hero_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    cols = st.columns(4)
    metrics = [
        ("1,348", T["patient_records"]),
        (f"{max(r['accuracy'] for r in model_results.values()) * 100:.1f}%", T["best_accuracy"]),
        ("3", T["models_trained"]),
        ("4", T["hyp_stages"]),
    ]
    for col, (val, lbl) in zip(cols, metrics):
        col.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    features = [
        ("🔬", T["feat_class_title"], T["feat_class_desc"]),
        ("📈", T["feat_risk_title"], T["feat_risk_desc"]),
        ("🥗", T["feat_life_title"], T["feat_life_desc"]),
        ("🤖", T["feat_ai_title"], T["feat_ai_desc"]),
        ("📊", T["feat_eda_title"], T["feat_eda_desc"]),
        ("⚡", T["feat_model_title"], T["feat_model_desc"]),
    ]
    rows = [features[i:i+3] for i in range(0, len(features), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            col.markdown(f"""
            <div class="feature-card">
                <div class="icon">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Scenario section
    st.markdown(f"### 🏥 {T['use_cases']}")
    tabs = st.tabs([T["uc_screening"], T["uc_clinical"], T["uc_research"]])
    scenarios = [
        T["uc_screening_desc"],
        T["uc_clinical_desc"],
        T["uc_research_desc"],
    ]
    for tab, text in zip(tabs, scenarios):
        with tab:
            st.markdown(text)


# ═════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ═════════════════════════════════════════════════════════
elif page_key == "eda":
    st.markdown(f"""
    <div class="hero-card" style="padding:2rem 2.5rem;">
        <h1 style="font-size:2rem;">{T['eda_title']}</h1>
        <p>{T['eda_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset overview
    cols = st.columns(4)
    cols[0].metric("Total Records", f"{len(df_raw):,}")
    cols[1].metric("Features", len(df_raw.columns) - 1)
    cols[2].metric("Target Classes", df_raw["Stages"].nunique())
    cols[3].metric("Duplicates Removed", "477")

    st.markdown("---")

    # ── Chart 1 & 2: Gender Distribution ──
    st.markdown(f"### {T['gender_dist']}")
    c1, c2 = st.columns(2)
    gender_counts = df_raw["Gender"].value_counts()
    with c1:
        fig = px.bar(
            x=gender_counts.index, y=gender_counts.values,
            color=gender_counts.index,
            color_discrete_map={"Male": pri, "Female": acc},
            labels={"x": "Gender", "y": "Count"},
        )
        fig.update_layout(template=chart_template, showlegend=False, height=380, paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(
            values=gender_counts.values, names=gender_counts.index,
            color_discrete_sequence=[pri, acc],
            hole=0.45,
        )
        fig.update_layout(template=chart_template, height=380, paper_bgcolor=bg_main if dark else None, font_color=text_main)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Chart 3: Hypertension Stages Distribution ──
    st.markdown(f"### {T['stages_dist']}")
    stages_counts = df_raw["Stages"].value_counts()
    color_map = {
        "NORMAL": "#10b981",
        "HYPERTENSION (Stage-1)": "#f59e0b",
        "HYPERTENSION (Stage-2)": "#ef4444",
        "HYPERTENSIVE CRISIS": "#991b1b",
    }
    fig = px.bar(
        x=stages_counts.index, y=stages_counts.values,
        color=stages_counts.index,
        color_discrete_map=color_map,
        labels={"x": "Stage", "y": "Count"},
    )
    fig.update_layout(template=chart_template, showlegend=False, height=400, paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Chart 4: Correlation Heatmap ──
    st.markdown(f"### {T['bp_corr']}")
    sys_map = {"100+": 105, "111 - 120": 115.5, "121 - 130": 125.5, "130+": 140}
    dia_map = {"70 - 80": 75, "81 - 90": 85.5, "91 - 100": 95.5, "100+": 110, "130+": 140}
    df_corr = pd.DataFrame({
        "Systolic (midpoint)": df_raw["Systolic"].map(sys_map),
        "Diastolic (midpoint)": df_raw["Diastolic"].map(dia_map),
    })
    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="YlOrRd", fmt=".3f", linewidths=1,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Blood Pressure Correlation", fontsize=13, fontweight="bold", color=mpl_text)
    fig.patch.set_facecolor(mpl_face)
    ax.set_facecolor(mpl_face)
    ax.tick_params(colors=mpl_tick)
    ax.xaxis.label.set_color(mpl_tick)
    ax.yaxis.label.set_color(mpl_tick)
    for text in ax.texts:
        text.set_color(mpl_text)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Chart 5: Medication vs Severity ──
    st.markdown(f"### {T['med_severity']}")
    fig = px.histogram(
        df_raw, x="TakeMedication", color="Severity",
        barmode="group",
        color_discrete_sequence=[pri, acc, theme["chart3"]],
        labels={"TakeMedication": "Takes Medication", "count": "Count"},
    )
    fig.update_layout(template=chart_template, height=400, paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Chart 6: Age Group vs Stages ──
    st.markdown(f"### {T['age_stages']}")
    fig = px.histogram(
        df_raw, x="Age", color="Stages",
        barmode="group",
        color_discrete_map=color_map,
        category_orders={"Age": ["18-34", "35-50", "51-64", "65+"]},
        labels={"Age": "Age Group"},
    )
    fig.update_layout(template=chart_template, height=420, paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Chart 7: Scatter Systolic vs Diastolic by Stage ──
    st.markdown(f"### {T['scatter_bp']}")
    df_scatter = df_raw.copy()
    df_scatter["Systolic_mid"] = df_scatter["Systolic"].map(sys_map)
    df_scatter["Diastolic_mid"] = df_scatter["Diastolic"].map(dia_map)
    fig = px.scatter(
        df_scatter, x="Systolic_mid", y="Diastolic_mid", color="Stages",
        color_discrete_map=color_map,
        labels={"Systolic_mid": "Systolic BP (midpoint)", "Diastolic_mid": "Diastolic BP (midpoint)"},
        opacity=0.7,
    )
    fig.update_layout(template=chart_template, height=450, paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════
elif page_key == "model":
    st.markdown(f"""
    <div class="hero-card" style="padding:2rem 2.5rem;">
        <h1 style="font-size:2rem;">{T['model_title']}</h1>
        <p>{T['model_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Accuracy Comparison ──
    st.markdown(f"### {T['accuracy_comp']}")
    model_names = list(model_results.keys())
    accuracies = [model_results[n]["accuracy"] * 100 for n in model_names]
    colors = [pri, acc, theme["chart3"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names, y=accuracies,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{a:.1f}%" for a in accuracies],
        textposition="outside",
        textfont=dict(size=16, color=text_main),
    ))
    fig.update_layout(
        template=chart_template, height=400,
        yaxis=dict(range=[0, 110], title="Accuracy (%)"),
        paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary Table ──
    st.markdown(f"### {T['perf_summary']}")
    summary_data = []
    for name in model_names:
        r = model_results[name]
        rep = r["report"]
        summary_data.append({
            "Model": name,
            "Accuracy (%)": f"{r['accuracy']*100:.1f}",
            "Macro Precision": f"{rep['macro avg']['precision']:.3f}",
            "Macro Recall": f"{rep['macro avg']['recall']:.3f}",
            "Macro F1": f"{rep['macro avg']['f1-score']:.3f}",
            "Weighted F1": f"{rep['weighted avg']['f1-score']:.3f}",
        })
    st.dataframe(pd.DataFrame(summary_data).set_index("Model"), use_container_width=True)

    # ── Per-model details ──
    st.markdown(f"### {T['detailed_reports']}")
    tabs = st.tabs(model_names)
    for tab, name in zip(tabs, model_names):
        with tab:
            r = model_results[name]

            # Classification report
            report_df = pd.DataFrame(r["report"]).T
            report_df = report_df.drop(["accuracy"], errors="ignore")
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

            # Confusion matrix
            st.markdown(f"**{T['conf_matrix']}**")
            cm = r["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=[stage_labels[i] for i in sorted(stage_labels)],
                        yticklabels=[stage_labels[i] for i in sorted(stage_labels)],
                        ax=ax, linewidths=0.5)
            ax.set_xlabel("Predicted", fontsize=11, color=mpl_text)
            ax.set_ylabel("Actual", fontsize=11, color=mpl_text)
            ax.set_title(f"{name} — {T['conf_matrix']}", fontsize=13, fontweight="bold", color=mpl_text)
            fig.patch.set_facecolor(mpl_face)
            ax.set_facecolor(mpl_face)
            ax.tick_params(colors=mpl_tick, labelsize=8)
            for text_obj in ax.texts:
                text_obj.set_color(mpl_text)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Model selection rationale ──
    st.markdown(f"### {T['model_rationale']}")
    best_name = max(model_results, key=lambda n: model_results[n]["accuracy"])
    st.info(f"**Selected Model: {best_name}** with {model_results[best_name]['accuracy']*100:.1f}% accuracy")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Why not 100% accuracy models?**
        - Perfect accuracy is a classic sign of **overfitting**
        - Models may memorize training patterns instead of learning generalizable features
        - Poor performance on new, unseen patient data
        - Risk of false confidence in clinical decisions
        """)
    with col2:
        st.markdown("""
        **Key selection criteria:**
        - ✅ Strong generalization capability
        - ✅ Balanced precision and recall
        - ✅ Robust across all hypertension stages
        - ✅ Clinically reliable predictions
        - ✅ High crisis recall (no missed emergencies)
        """)


# ═════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════════════════
elif page_key == "predict":
    st.markdown(f"""
    <div class="hero-card" style="padding:2rem 2.5rem;">
        <h1 style="font-size:2rem;">{T['predict_title']}</h1>
        <p>{T['predict_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ──
    with st.form("prediction_form"):
        st.markdown("#### 👤 Demographics")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        age = c2.selectbox("Age Group", ["18-34", "35-50", "51-64", "65+"])

        st.markdown("#### 🏥 Medical History")
        c1, c2, c3, c4 = st.columns(4)
        history = c1.selectbox("Family History", ["No", "Yes"])
        patient = c2.selectbox("Current Patient", ["No", "Yes"])
        medication = c3.selectbox("Takes Medication", ["No", "Yes"])
        diagnosed = c4.selectbox("When Diagnosed", ["<1 Year", "1 - 5 Years", ">5 Years"])

        st.markdown("#### 🩻 Symptoms")
        c1, c2, c3, c4 = st.columns(4)
        severity = c1.selectbox("Severity", ["Mild", "Moderate", "Severe"])
        breath = c2.selectbox("Shortness of Breath", ["No", "Yes"])
        visual = c3.selectbox("Visual Changes", ["No", "Yes"])
        nose = c4.selectbox("Nose Bleeding", ["No", "Yes"])

        st.markdown("#### 💓 Vital Signs & Lifestyle")
        c1, c2, c3 = st.columns(3)
        systolic = c1.selectbox("Systolic BP Range", ["100+", "111 - 120", "121 - 130", "130+"])
        diastolic = c2.selectbox("Diastolic BP Range", ["70 - 80", "81 - 90", "91 - 100", "100+", "130+"])
        diet = c3.selectbox("Controlled Diet", ["No", "Yes"])

        submitted = st.form_submit_button("🔍  Analyze Risk", use_container_width=True)

    if submitted:
        # Encode inputs
        patient_info = {
            "Gender": gender, "Age": age, "History": history, "Patient": patient,
            "TakeMedication": medication, "Severity": severity, "BreathShortness": breath,
            "VisualChanges": visual, "NoseBleeding": nose, "Whendiagnoused": diagnosed,
            "Systolic": systolic, "Diastolic": diastolic, "ControlledDiet": diet,
        }
        input_encoded = np.array([[
            label_mappings["Gender"][gender],
            label_mappings["Age"][age],
            label_mappings["History"][history],
            label_mappings["Patient"][patient],
            label_mappings["TakeMedication"][medication],
            label_mappings["Severity"][severity],
            label_mappings["BreathShortness"][breath],
            label_mappings["VisualChanges"][visual],
            label_mappings["NoseBleeding"][nose],
            label_mappings["Whendiagnoused"][diagnosed],
            label_mappings["Systolic"][systolic],
            label_mappings["Diastolic"][diastolic],
            label_mappings["ControlledDiet"][diet],
        ]])
        input_scaled = scaler.transform(input_encoded)

        # Use best model for prediction
        best_model_name = max(model_results, key=lambda n: model_results[n]["accuracy"])
        best_model = model_results[best_model_name]["model"]
        prediction = best_model.predict(input_scaled)[0]
        probabilities = best_model.predict_proba(input_scaled)[0]
        confidence = probabilities[prediction] * 100
        stage_name = stage_labels[prediction]

        st.markdown("---")
        st.markdown("### 📋 Risk Assessment Results")

        # Risk card
        risk_class = {0: "risk-normal", 1: "risk-stage1", 2: "risk-stage2", 3: "risk-crisis"}
        risk_emoji = {0: "✅", 1: "⚠️", 2: "🔴", 3: "🚨"}
        risk_action = {
            0: "Maintain healthy lifestyle. Schedule routine check-ups every 6–12 months.",
            1: "Lifestyle modifications recommended. Monitor BP weekly. Consult physician within 30 days.",
            2: "Medical intervention recommended. Schedule appointment within 1 week. Daily BP monitoring required.",
            3: "⚡ IMMEDIATE MEDICAL ATTENTION REQUIRED. Seek emergency care if experiencing severe symptoms.",
        }

        st.markdown(f"""
        <div class="{risk_class[prediction]}">
            <div class="risk-title">{risk_emoji[prediction]} {T['risk_complete']}</div>
            <div class="risk-stage">{T['classification']}: {stage_name}</div>
            <p style="margin-top:0.8rem;font-size:1.05rem;">{T['confidence']}: <strong>{confidence:.1f}%</strong> &nbsp;|&nbsp; {T['model_label']}: <strong>{best_model_name}</strong></p>
            <p style="margin-top:0.5rem;font-size:0.95rem;">{risk_action[prediction]}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability distribution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### {T['prob_dist']}")
        prob_df = pd.DataFrame({
            "Stage": [stage_labels[i] for i in range(len(probabilities))],
            "Probability (%)": probabilities * 100,
        })
        fig = px.bar(
            prob_df, x="Stage", y="Probability (%)",
            color="Stage",
            color_discrete_map=color_map if 'color_map' in dir() else None,
            text=[f"{p:.1f}%" for p in probabilities * 100],
        )
        color_map_prob = {
            "NORMAL": "#10b981", "HYPERTENSION (Stage-1)": "#f59e0b",
            "HYPERTENSION (Stage-2)": "#ef4444", "HYPERTENSIVE CRISIS": "#991b1b",
        }
        fig.for_each_trace(lambda t: t.update(marker_color=color_map_prob.get(t.name, pri)))
        fig.update_layout(
            template=chart_template, height=350, showlegend=False,
            paper_bgcolor=bg_main if dark else None, plot_bgcolor=bg_main if dark else None, font_color=text_main,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # ── AI Health Insights ──
        st.markdown("---")
        st.markdown(f"### {T['ai_insights']}")
        with st.spinner(T['ai_generating']):
            insight = get_groq_insight(stage_name, confidence, patient_info)
        st.markdown(f"""<div class="ai-insight"><h3>{T['ai_assessment']}</h3></div>""", unsafe_allow_html=True)
        st.markdown(insight)


# ═════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════
elif page_key == "about":
    st.markdown(f"""
    <div class="hero-card" style="padding:2rem 2.5rem;">
        <h1 style="font-size:2rem;">{T['about_title']}</h1>
        <p>{T['about_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-card">
        <h3 style="color:{pri};">{T['project_overview']}</h3>
        <p style="color:{text_secondary};line-height:1.8;">
        {T['overview_text']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="about-card">
            <h3 style="color:{pri};">{T['tech_used']}</h3>
            <ul style="color:{text_secondary};line-height:2;">
                <li><strong>Python</strong> — Core programming language</li>
                <li><strong>Streamlit</strong> — Web application framework</li>
                <li><strong>Pandas & NumPy</strong> — Data manipulation</li>
                <li><strong>Scikit-learn</strong> — ML model training</li>
                <li><strong>XGBoost</strong> — Gradient boosting</li>
                <li><strong>Plotly & Seaborn</strong> — Visualizations</li>
                <li><strong>Groq API</strong> — AI text generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="about-card">
            <h3 style="color:{pri};">{T['future_impl']}</h3>
            <ul style="color:{text_secondary};line-height:2;">
                <li><strong>EMR Integration</strong> — Link to Electronic Health Records</li>
                <li><strong>Multi-Stage Output</strong> — More detailed clinical guidance</li>
                <li><strong>Wearable Data</strong> — Real-time BP monitor analysis</li>
                <li><strong>XAI for Trust</strong> — SHAP-based explanations</li>
                <li><strong>Mobile App</strong> — On-the-go risk assessment</li>
                <li><strong>Dataset Expansion</strong> — Better generalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-card" style="text-align:center;">
        <h3 style="color:{pri};">👥 Team GenV</h3>
        <p style="color:{text_secondary};font-size:0.95rem;margin-bottom:1.2rem;">
            3rd Year | Department of ECE<br>
            <strong style="color:{pri};">Santhiram Engineering College (Autonomous), Nandyal</strong>
        </p>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0.8rem;">
            <span style="background:linear-gradient(135deg,{pri},{sec});color:white;padding:0.5rem 1.2rem;border-radius:50px;font-weight:600;font-size:0.9rem;">D. Yogendra Sai Teja Babu</span>
            <span style="background:linear-gradient(135deg,{acc},#f472b6);color:white;padding:0.5rem 1.2rem;border-radius:50px;font-weight:600;font-size:0.9rem;">C. V. Karimulla</span>
            <span style="background:linear-gradient(135deg,#10b981,#34d399);color:white;padding:0.5rem 1.2rem;border-radius:50px;font-weight:600;font-size:0.9rem;">G. Sumanth</span>
            <span style="background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#1e293b;padding:0.5rem 1.2rem;border-radius:50px;font-weight:600;font-size:0.9rem;">M. Sri Ravindranath</span>
            <span style="background:linear-gradient(135deg,#6366f1,#818cf8);color:white;padding:0.5rem 1.2rem;border-radius:50px;font-weight:600;font-size:0.9rem;">G. Surya Prakash</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-card" style="text-align:center; margin-top:1rem;">
        <p style="color:#ef4444;font-weight:600;font-size:0.9rem;">
        {T['disclaimer']}
        </p>
    </div>
    """, unsafe_allow_html=True)
