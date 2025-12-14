import streamlit as st

def apply_theme_and_header():
    st.set_page_config(page_title="Web Data Mining", layout="wide")
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(-45deg, #0F2027, #203A43, #2C5364, #1B2735);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #F8F9FA;
        padding-bottom: 2rem;
    }
    @keyframes gradientBG { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    .title { text-align:center; color:#00B4D8; font-size:34px; font-weight:700; }
    .subtitle { text-align:center; color:#A8DADC; margin-bottom:18px; }
    </style>
    """, unsafe_allow_html=True)

def footer():
    st.markdown("""
    <hr style="border:1px solid #00B4D8; margin-top:28px;">
    <p style="text-align:center; color:gray;"></b></p>
    """, unsafe_allow_html=True)
