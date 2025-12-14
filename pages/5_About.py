import streamlit as st
from src.ui_helpers import apply_theme_and_header, footer

apply_theme_and_header()
st.markdown('<div class="title">About the Developer</div>', unsafe_allow_html=True)

st.markdown("""
### MCA Project — Web Data Mining & Terrorism Analysis
**Developed by:** Thanusha  and Reeoan   
**University:** Presidency University     
**Year:** 2025  

####  Tools
- Python • Pandas • Scikit-learn • NLTK • VADER  
- Streamlit • Plotly • WordCloud  
- Tweepy (Twitter API)

####  Objective
Analyze real-time Twitter data related to terrorism using NLP & ML,
providing insights via clustering, sentiment analysis, and topic modeling.
""")

footer()
