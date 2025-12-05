import streamlit as st
from predict import predict_job

# --- Page Config ---
st.set_page_config(page_title="JobGuard: Fake Job Classifier", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    /* Overall page background and text */
    .stApp {
        background-color: #262626;  /* dark grey background */
        color: white;
    }

    /* General text, headers, markdown */
    .stMarkdown, h1, h2, h3, h4, h5, h6, p, label {
        color: white;
    }

    /* Input boxes and text areas */
    input, textarea {
        background-color: #3b3b3b !important;
        color: white !important;
        border: 1px solid #555555 !important;
    }

    /* Buttons */
    .stButton>button {
        padding: 10px 28px; /* increase padding for bigger button */
        background-color: #3366ff; /* button color */
        color: white; /* text color */
        font-size: 18px; /* larger font */
        font-weight: bold;
        border-radius: 10px; /* rounded corners */
        border: none;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3); /* subtle shadow */
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #3333ff;
        transform: scale(1.05);
    }

    /* Checkboxes and radio buttons */
    .stCheckbox>div, .stRadio>div {
        color: white;
    }
    input[type="checkbox"] {
        accent-color: #f39c12;  /* optional: checkbox highlight color */
    }
    input[type="radio"] {
        accent-color: #f39c12;
    }

    /* Sliders */
    .stSlider>div>div>div>input {
        background-color: #3b3b3b !important;
        color: white !important;
    }
    .stSlider>div>div>div>div {
        background-color: #444444 !important;
    }

    /* Tables / DataFrames */
    .stDataFrame, .stTable {
        background-color: #3b3b3b;
        color: white;
    }

    /* Optional: expanders */
    .stExpanderHeader {
        background-color: #3b3b3b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Tagline ---
st.title("🛡️ JobGuard: Fake Job Classifier")
st.markdown("### Spot fraudulent job postings before you apply.")

st.write("---")

# --- Input Fields ---
title = st.text_input("Job Title*")
company = st.text_area("Company Profile", height=100)
description = st.text_area("Job Description*", height=100)
requirements = st.text_area("Job Requirements", height=100)
benefits = st.text_area("Benefits Provided", height=100)
salary = st.text_input("Salary (e.g., 50000 or 50000-60000)")

# --- Classify Button ---
if st.button("Classify"):
    if not title.strip():
        st.warning("⚠️ Please fill in the job title.")
    elif not description.strip():
        st.warning("⚠️ Please fill in the job description.")
    else:
        title = title.strip() if title else ""
        description = description.strip() if description else ""
        company = company.strip() if company else ""
        requirements = requirements.strip() if requirements else ""
        benefits = benefits.strip() if benefits else ""
        salary = salary.strip() if salary else ""
        
        with st.spinner("Analyzing..."):
            score, result = predict_job(title, company, description, requirements, benefits, salary)

        st.write("---")
        if result == 0:
            st.success("✅ This job posting seems **REAL**.")
        else:
            st.error("🚨 This job posting appears to be **FAKE**.")