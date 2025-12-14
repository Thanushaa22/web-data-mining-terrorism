"""
Web Data Mining – Terrorism Tweet Analysis
Single-file Streamlit App (Stable & Deployable)
Developer: Thanusha (MCA Project)
"""

# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Optional Plotly
try:
    import plotly.express as px
    PLOTLY = True
except:
    PLOTLY = False


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Web Data Mining – Tweet Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================== BASIC STYLE ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #EAF6FF;
}
h1, h2, h3 {
    color: #00E6FF;
}
</style>
""", unsafe_allow_html=True)

# ---------- CSS / Styling ---------- # background image (transparent PNG or subtle texture) - replace URL if you prefer
BG_IMG = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST2Rs_vAVQSuYVs3xAvlmYYtJHde09GCg91Q&s"

st.markdown( f""" <style> /* animated gradient overlay */ .stApp {{ background-image: linear-gradient(135deg, rgba(10,25,47,0.85), rgba(20,40,63,0.75)), url('{BG_IMG}'); background-size: cover; background-attachment: fixed; color: #E6F2F8; }} /* hero card */ .hero {{ padding: 24px; border-radius: 14px; background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); box-shadow: 0 6px 30px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.03); }} .neon-btn {{ display:inline-block; margin:6px; padding:10px 16px; border-radius:10px; color:#00F5FF; text-decoration:none; border: 1px solid rgba(0,245,255,0.2); background: rgba(0,245,255,0.02); box-shadow: 0 4px 14px rgba(0,245,255,0.04); transition: all 0.18s ease-in-out; }} .neon-btn:hover {{ transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,245,255,0.08); background: rgba(0,245,255,0.035); }} .small-muted {{ color: #bcd; font-size:12px; }} .stat-card {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding: 12px; border-radius: 10px; text-align:center; border: 1px solid rgba(255,255,255,0.02); }} </style> """, unsafe_allow_html=True, )
# ================== HELPERS ==================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

analyzer = SentimentIntensityAnalyzer()

def sentiment_label(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return " ".join(text.split()).lower()

def build_wordcloud(texts):
    if not texts:
        return None
    wc = WordCloud(width=900, height=400, background_color="white",
                   colormap="viridis").generate(" ".join(texts))
    return wc


# ================== LOAD DATA ==================
tweets = load_csv("data/twitter_tweets_clean.csv")
clusters = load_csv("data/twitter_clusters.csv")
sentiments = load_csv("data/twitter_sentiment.csv")

if not tweets.empty:
    if "clean_text" not in tweets:
        tweets["clean_text"] = tweets["text"].apply(clean_text)

    if not sentiments.empty and "sentiment" in sentiments:
        tweets["sentiment"] = sentiments["sentiment"]
    else:
        tweets["sentiment"] = tweets["clean_text"].apply(sentiment_label)

    if not clusters.empty and "cluster" in clusters:
        tweets["cluster"] = clusters["cluster"]
    else:
        tweets["cluster"] = 0

    if "created_at" in tweets:
        tweets["created_at"] = pd.to_datetime(
            tweets["created_at"], errors="coerce"
        ).dt.tz_localize(None)


# ================== SIDEBAR ==================
st.sidebar.title("Filters")

cluster_list = sorted(tweets["cluster"].unique()) if not tweets.empty else []
sentiment_list = ["Positive", "Neutral", "Negative"]

selected_clusters = st.sidebar.multiselect(
    "Select Clusters",
    options=cluster_list,
    default=cluster_list,
    key="cluster_filter"
)

selected_sentiments = st.sidebar.multiselect(
    "Select Sentiment",
    options=sentiment_list,
    default=sentiment_list,
    key="sentiment_filter"
)

search_keyword = st.sidebar.text_input(
    "Search Keyword",
    key="search_filter"
)

start_date = end_date = None
if not tweets.empty and "created_at" in tweets:
    min_d = tweets["created_at"].dt.date.min()
    max_d = tweets["created_at"].dt.date.max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=[min_d, max_d],
        min_value=min_d,
        max_value=max_d,
        key="date_filter"
    )

    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range


# ================== NAVIGATION ==================
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dashboard", "Insights", "About"],
    key="page_nav"
)


# ================== FILTER DATA ==================
df = tweets.copy()

if selected_clusters:
    df = df[df["cluster"].isin(selected_clusters)]

if selected_sentiments:
    df = df[df["sentiment"].isin(selected_sentiments)]

if search_keyword:
    df = df[df["clean_text"].str.contains(search_keyword, case=False, na=False)]

if start_date and end_date and "created_at" in df:
    df = df[
        (df["created_at"] >= pd.to_datetime(start_date)) &
        (df["created_at"] <= pd.to_datetime(end_date))
    ]


# ================== HOME ==================
if page == "Home":
    st.title("Web Data Mining – Terrorism Tweet Analysis")
    st.write("""
    This system analyzes terrorism-related tweets using:
    - Text preprocessing & cleaning
    - Sentiment analysis (VADER)
    - Clustering (K-Means)
    - Visual analytics
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tweets", len(tweets))
    c2.metric("Clusters", tweets["cluster"].nunique() if not tweets.empty else 0)
    c3.metric("Positive %",
              f"{(tweets['sentiment']=='Positive').mean()*100:.1f}%" if not tweets.empty else "0%")


# ================== DASHBOARD ==================
elif page == "Dashboard":
    st.title("Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Tweets", len(df))
    col2.metric("Clusters", df["cluster"].nunique() if not df.empty else 0)
    col3.metric("Positive %",
                f"{(df['sentiment']=='Positive').mean()*100:.1f}%" if not df.empty else "0%")

    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="sentiment",
                  order=sentiment_list, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster vs Sentiment")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="cluster", hue="sentiment",
                  palette="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("WordCloud")
    wc = build_wordcloud(df["clean_text"].tolist())
    if wc:
        fig_wc, ax_wc = plt.subplots(figsize=(10,4))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    st.subheader("Sample Tweets")
    st.dataframe(df[["clean_text", "sentiment", "cluster"]].head(50))

    st.download_button(
        "⬇ Download Filtered CSV",
        df.to_csv(index=False).encode("utf-8"),
        "filtered_tweets.csv",
        "text/csv"
    )


# ================== INSIGHTS ==================
elif page == "Insights":
    st.title("Insights")

    st.write("### Overall Sentiment Share")
    share = tweets["sentiment"].value_counts(normalize=True) * 100
    for k, v in share.items():
        st.metric(k, f"{v:.1f}%")

    st.write("### Top Keywords per Cluster")
    for c in sorted(tweets["cluster"].unique()):
        words = tweets[tweets["cluster"] == c]["clean_text"]
        words = " ".join(words).split()
        common = pd.Series(words).value_counts().head(8).index.tolist()
        st.write(f"**Cluster {c}:** {', '.join(common)}")


# ================== ABOUT ==================
elif page == "About":
    st.title("ℹ About")
    st.write("""
    **Web Data Mining – Terrorism Tweet Analysis**

    - Built using Python & Streamlit
    - Sentiment Analysis: VADER
    - Clustering: K-Means
    - MCA Final Project (2025)
    """)

