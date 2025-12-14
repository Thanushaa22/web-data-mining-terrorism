# app.py
"""
Web Data Mining - Tweet Analysis
Single-file Streamlit app with multiple internal pages (Home, Dashboard, Live Tweets, Insights, About)
Designed to be robust (no timezone comparison errors), visually appealing, and functional.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Try import Plotly(optional)
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

# ---------- Page config ----------
st.set_page_config(page_title="Web Data Mining — Tweet Analysis", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS / Styling ----------
# background image (transparent PNG or subtle texture) - replace URL if you prefer
BG_IMG = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST2Rs_vAVQSuYVs3xAvlmYYtJHde09GCg91Q&s"


st.markdown(
    f"""
    <style>
    /* animated gradient overlay */
    .stApp {{
      background-image: linear-gradient(135deg, rgba(10,25,47,0.85), rgba(20,40,63,0.75)), url('{BG_IMG}');
      background-size: cover;
      background-attachment: fixed;
      color: #E6F2F8;
    }}
    /* hero card */
    .hero {{
      padding: 24px;
      border-radius: 14px;
      background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      box-shadow: 0 6px 30px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.03);
    }}
    .neon-btn {{
      display:inline-block;
      margin:6px;
      padding:10px 16px;
      border-radius:10px;
      color:#00F5FF;
      text-decoration:none;
      border: 1px solid rgba(0,245,255,0.2);
      background: rgba(0,245,255,0.02);
      box-shadow: 0 4px 14px rgba(0,245,255,0.04);
      transition: all 0.18s ease-in-out;
    }}
    .neon-btn:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 30px rgba(0,245,255,0.08);
      background: rgba(0,245,255,0.035);
    }}
    .small-muted {{ color: #bcd; font-size:12px; }}
    .stat-card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      padding: 12px;
      border-radius: 10px;
      text-align:center;
      border: 1px solid rgba(255,255,255,0.02);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
@st.cache_data
def load_csv_safe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

analyzer = SentimentIntensityAnalyzer()

def sentiment_label(text: str) -> str:
    s = analyzer.polarity_scores(str(text))["compound"]
    return "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")

def make_wordcloud(texts, width=800, height=300, background_color="black"):
    if not texts:
        return None
    text = " ".join(texts)
    wc = WordCloud(width=width, height=height, background_color=background_color, colormap="cool").generate(text)
    return wc

def safe_merge_tweets_clusters(tweets_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tweets & clusters robustly:
     - if clusters has 'id', merge on id
     - else if lengths match, attach by position
     - else set cluster=0
    """
    df = tweets_df.copy()
    if clusters_df is None or clusters_df.empty:
        df["cluster"] = df.get("cluster", 0)
        return df

    if "id" in clusters_df.columns and "id" in df.columns:
        clusters_df = clusters_df.rename(columns=lambda c: c.strip())
        df = df.merge(clusters_df[["id", "cluster"]], how="left", on="id")
        df["cluster"] = df["cluster"].fillna(0).astype(int)
        return df
    # if single column with cluster and same length
    if len(clusters_df) == len(df):
        if "cluster" in clusters_df.columns:
            df["cluster"] = clusters_df["cluster"].values
        else:
            df["cluster"] = clusters_df.iloc[:, 0].values
        return df
    # fallback
    df["cluster"] = df.get("cluster", 0)
    return df

def tz_safe_to_date(series: pd.Series) -> pd.Series:
    """Convert datetime series to tz-naive dates safely."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # older pandas may raise; ensure naive
        pass
    return s

# ---------- Load datasets ----------
TWEETS_CSV = "data/twitter_tweets_clean.csv"
CLUSTERS_CSV = "data/twitter_clusters.csv"
FEATURES_CSV = "data/twitter_features.csv"
SENTIMENT_CSV = "data/twitter_sentiment.csv"

tweets = load_csv_safe(TWEETS_CSV)
clusters = load_csv_safe(CLUSTERS_CSV)
features = load_csv_safe(FEATURES_CSV)
sentiment_df = load_csv_safe(SENTIMENT_CSV)

# Merge safely
merged = safe_merge_tweets_clusters(tweets, clusters)

# Ensure clean_text present
if "clean_text" not in merged.columns and "text" in merged.columns:
    def simple_clean(s):
        if not isinstance(s, str): return ""
        s = re.sub(r"http\S+|www\S+", "", s)
        s = re.sub(r"@\w+", "", s)
        s = re.sub(r"#\w+", "", s)
        s = re.sub(r"[^A-Za-z\s]", " ", s)
        return " ".join(s.split()).lower()
    merged["clean_text"] = merged["text"].fillna("").apply(simple_clean)

# Ensure sentiment present
if "sentiment" not in merged.columns:
    if not sentiment_df.empty and "id" in sentiment_df.columns:
        merged = merged.merge(sentiment_df[["id", "sentiment"]], how="left", on="id")
        merged["sentiment"] = merged["sentiment"].fillna(merged.get("sentiment", np.nan))
    if "sentiment" not in merged.columns or merged["sentiment"].isnull().all():
        if "clean_text" in merged.columns:
            merged["sentiment"] = merged["clean_text"].apply(sentiment_label)
        else:
            merged["sentiment"] = "Neutral"

# created_at safe
if "created_at" in merged.columns:
    merged["created_at"] = tz_safe_to_date(merged["created_at"])
    merged = merged.dropna(subset=["created_at"])

# ---------- App-level navigation (single-file multi-page) ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Dashboard", "Live Tweets", "Insights", "About"))

# ---------- Shared sidebar filters (used by Dashboard/Insights) ----------
st.sidebar.markdown("---")
cluster_options = sorted(merged["cluster"].dropna().unique().tolist()) if "cluster" in merged.columns else []
selected_clusters = st.sidebar.multiselect("Clusters", options=cluster_options, default=cluster_options if cluster_options else [])
sentiment_options = ["Positive", "Neutral", "Negative"]
selected_sentiments = st.sidebar.multiselect("Sentiment", options=sentiment_options, default=sentiment_options)
search_text = st.sidebar.text_input("Search text (free text)")

# Date picker safe (Streamlit Cloud fix)
start_date = end_date = None
if "created_at" in merged.columns and not merged.empty:
    min_date = merged["created_at"].dt.date.min()
    max_date = merged["created_at"].dt.date.max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=[min_date, max_date],  # MUST be list, NOT tuple
        min_value=min_date,
        max_value=max_date
    )

    # Handle Streamlit's return type
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range

# ---------- Home Page ----------
if page == "Home":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#00E6FF;'>Web Data Mining — Terrorism Tweet Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted'>Real-time monitoring, clustering & sentiment analysis of tweets related to terrorism and extremism.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("""
        ### What this project does
        - Collects tweets using Twitter API (Tweepy)  
        - Cleans and preprocesses tweets (regex, tokenization)  
        - Extracts TF-IDF features and clusters tweets (K-Means / embedding options)  
        - Analyzes sentiment using VADER (and supports transformer models later)  
        - Visualizes everything in an interactive dashboard with export options
        """)
        st.markdown("### Quick Actions")
        st.markdown(f"<a class='neon-btn' href='#' onclick='document.querySelector(\"[data-testid=stSidebar] input\").click()'>Open Filters</a>", unsafe_allow_html=True)
        st.markdown(f"<a class='neon-btn' href='#' onclick='window.scrollTo(0,document.body.scrollHeight);'>Go to Dashboard</a>", unsafe_allow_html=True)

    with c2:
        st.markdown("### Live Summary")
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.metric("Total Tweets", len(merged))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.metric("Clusters", merged["cluster"].nunique() if "cluster" in merged.columns else 0)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How to use")
    st.write("""
    1. Use the sidebar to filter by cluster, sentiment or date range.  
    2. Visit **Live Tweets** to fetch fresh tweets (you must set TWITTER_BEARER_TOKEN).  
    3. Download filtered data from the Dashboard or Live fetch results.  
    """)

# ---------- Dashboard Page ----------
elif page == "Dashboard":
    st.markdown("<h2 style='color:#00E6FF'>Dashboard</h2>", unsafe_allow_html=True)
    # apply filters
    df_view = merged.copy()
    if selected_clusters:
        df_view = df_view[df_view["cluster"].isin(selected_clusters)]
    if selected_sentiments:
        df_view = df_view[df_view["sentiment"].isin(selected_sentiments)]
    if search_text:
        df_view = df_view[df_view["clean_text"].str.contains(search_text, case=False, na=False)]
    if start_date and end_date and "created_at" in df_view.columns:
        # ensure tz-naive and compare using date range
        df_view["created_at"] = tz_safe_to_date(df_view["created_at"])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df_view = df_view[(df_view["created_at"] >= start_dt) & (df_view["created_at"] < end_dt)]

    # metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Tweets", len(df_view))
    col2.metric("Unique Clusters", int(df_view["cluster"].nunique()) if not df_view.empty else 0)
    pospct = df_view["sentiment"].eq("Positive").mean() * 100 if not df_view.empty else 0
    col3.metric("Positive %", f"{pospct:.1f}%")
    col4.metric("Sample Date Range", f"{start_date} → {end_date}" if start_date and end_date else "All")

    st.markdown("### Sentiment Distribution")
    if df_view.empty:
        st.info("No data to display for selected filters.")
    else:
        if PLOTLY:
            fig = px.histogram(df_view, x="sentiment", color="sentiment", text_auto=True,
                               color_discrete_map={"Positive":"#2ECC71","Neutral":"#F1C40F","Negative":"#E74C3C"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            df_view["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    if "cluster" in df_view.columns and not df_view.empty:
        st.markdown("### Cluster-wise Sentiment")
        if PLOTLY:
            fig2 = px.histogram(df_view, x="cluster", color="sentiment", barmode="group")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write(df_view.groupby(["cluster","sentiment"]).size().unstack(fill_value=0))

    # Trend over time
    if "created_at" in df_view.columns and not df_view.empty:
        st.markdown("### Tweet Frequency Over Time")
        df_time = df_view.copy()
        df_time["date"] = pd.to_datetime(df_time["created_at"]).dt.date
        trend = df_time.groupby("date").size().reset_index(name="count")
        if PLOTLY:
            fig3 = px.line(trend, x="date", y="count", markers=True)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.plot(trend["date"], trend["count"], marker="o")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Wordcloud
    st.markdown("### WordCloud (filtered)")
    texts = df_view["clean_text"].dropna().astype(str).tolist()
    wc = make_wordcloud(texts, background_color="white")
    if wc is not None:
        fig_wc, ax_wc = plt.subplots(figsize=(10, 3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("No text available to build a wordcloud for the current filters.")

    # table + download
    st.markdown("### Sample Tweets")
    st.dataframe(df_view[["created_at","clean_text","sentiment","cluster"]].head(200))
    if not df_view.empty:
        st.download_button("⬇ Download filtered CSV", df_view.to_csv(index=False).encode("utf-8"), file_name="filtered_tweets.csv")

# ---------- Live Tweets Page ----------
elif page == "Live Tweets":
    st.markdown("<h2 style='color:#00E6FF'>Live Tweets</h2>", unsafe_allow_html=True)
    st.write("Fetch fresh tweets using your Twitter Bearer Token (kept secret).")

    query = st.text_input("Query", value="terrorism OR extremist")
    max_results = st.slider("Max tweets to fetch", 10, 100, 50)
    fetch_btn = st.button("Fetch live tweets")

    # fetch helper (not cached while debugging - caching may hide token changes)
    def fetch_live(query, max_results, bearer_token):
        try:
            from src.collect.twitter_collector import collect_tweets
        except Exception as e:
            st.error("Collector not found: ensure src/collect/twitter_collector.py exists and defines collect_tweets().")
            st.exception(e)
            return pd.DataFrame()
        try:
            df = collect_tweets(query=query, max_results=max_results, bearer_token=bearer_token)
            return df
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            return pd.DataFrame()

    # read token from secrets or environment
    bearer = None
    try:
        bearer = st.secrets.get("TWITTER_BEARER_TOKEN", None)
    except Exception:
        bearer = None
    if not bearer:
        bearer = os.getenv("TWITTER_BEARER_TOKEN")

    if not bearer:
        st.warning("Twitter bearer token not found. Add it to .streamlit/secrets.toml or set environment variable TWITTER_BEARER_TOKEN.")
        st.info("If you just added the env var, open a new terminal and restart Streamlit.")
    if fetch_btn:
        if not bearer:
            st.error("Cannot fetch — no bearer token.")
        else:
            with st.spinner("Fetching tweets..."):
                new_df = fetch_live(query, max_results, bearer)
                if new_df is None or new_df.empty:
                    st.warning("No tweets returned for that query.")
                else:
                    # clean + sentiment
                    new_df["clean_text"] = new_df["text"].fillna("").astype(str).str.replace(r"http\S+|www\S+|@\w+|#\w+", "", regex=True).str.replace(r"[^A-Za-z\s]","",regex=True).str.lower()
                    new_df["sentiment"] = new_df["clean_text"].apply(sentiment_label)
                    st.success(f"Fetched {len(new_df)} tweets")
                    st.dataframe(new_df[["created_at","clean_text","sentiment","retweets","likes"]].head(500))
                    st.download_button("⬇ Download fetched tweets", new_df.to_csv(index=False).encode("utf-8"), file_name="fetched_tweets.csv")

# ---------- Insights Page ----------
elif page == "Insights":
    st.markdown("<h2 style='color:#00E6FF'>Insights & Observations</h2>", unsafe_allow_html=True)
    if merged.empty:
        st.warning("No data to compute insights. Run collection & preprocessing first.")
    else:
        # sentiment share
        share = merged["sentiment"].value_counts(normalize=True).mul(100).round(1)
        st.metric("Positive", f"{share.get('Positive',0)}%")
        st.metric("Neutral", f"{share.get('Neutral',0)}%")
        st.metric("Negative", f"{share.get('Negative',0)}%")

        st.markdown("### Top keywords (overall)")
        def top_words(texts, topn=15):
            stop = set(["the","and","to","of","a","in","is","for","on","that","this","it","with","as"])
            cnt={}
            for t in texts:
                for w in re.findall(r"\b[a-z]{3,}\b", t.lower()):
                    if w in stop: continue
                    cnt[w]=cnt.get(w,0)+1
            return sorted(cnt.items(), key=lambda x:x[1], reverse=True)[:topn]
        overall = top_words(merged["clean_text"].dropna().astype(str).tolist())
        st.write(", ".join([w for w,_ in overall]))

        st.markdown("### Top keywords per cluster")
        for c in sorted(merged["cluster"].unique())[:8]:
            texts = merged[merged["cluster"]==c]["clean_text"].dropna().astype(str).tolist()
            tw = top_words(texts, topn=10)
            st.write(f"Cluster {c}: " + ", ".join([w for w,_ in tw]))

# ---------- About Page ----------
elif page == "About":
    st.markdown("<h2 style='color:#00E6FF'> About This Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    **Web Data Mining — Terrorism Tweet Analysis**  
    Built with Python, Streamlit, Tweepy, Scikit-learn, VADER and basic NLP tools.

    **Features**
    - Live tweet collection (Twitter API)  
    - Text cleaning, TF-IDF and clustering (K-Means)  
    - Sentiment analysis (VADER)  
    - Interactive visualization, wordclouds, CSV export

    **Developer:** Thanusha — MCA Project (2025)
    """)
    st.markdown("### Quick links")
    st.write("- GitHub repo: add your project's GitHub URL here")
    st.markdown("---")
    st.write("Tip: to enable live fetch create `.streamlit/secrets.toml` with `TWITTER_BEARER_TOKEN = \"<your token>\"` or set the environment variable `TWITTER_BEARER_TOKEN` in your terminal/session.")

# ---------- Footer ----------
st.markdown("""
<hr style="border:1px solid rgba(255,255,255,0.06);">
<p style="text-align:center; color:#bcd;"></p>
""", unsafe_allow_html=True)
