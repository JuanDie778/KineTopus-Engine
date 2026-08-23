import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Modern Card Style for Metrics */
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(0, 255, 204, 0.4);
            transform: translateY(-2px);
        }

        /* Upload Area Styling */
        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 1rem;
            background-color: rgba(255, 255, 255, 0.02);
        }

        /* Button Styling */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        
        /* Badges & Tags */
        .quant-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            background: rgba(0, 255, 204, 0.12);
            color: #00ffcc;
            border: 1px solid rgba(0, 255, 204, 0.3);
        }
        .quant-badge-secondary {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            background: rgba(255, 153, 0, 0.12);
            color: #ff9900;
            border: 1px solid rgba(255, 153, 0, 0.3);
        }

        /* Expander Header */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        </style>
    """, unsafe_allow_html=True)
