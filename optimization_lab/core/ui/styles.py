import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Modern Card Style for Metrics */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Upload Area Styling */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #4b5563;
            border-radius: 10px;
            padding: 2rem;
            background-color: #fafafa;
        }

        /* Button Styling */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Header Styling */
        h1 {
            color: #1e3a8a; /* Dark Blue */
            font-weight: 800;
        }
        
        h2, h3 {
            color: #374151;
        }

        </style>
    """, unsafe_allow_html=True)
