import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# ----------------------------------------------
# Page Setup
# ----------------------------------------------
st.set_page_config(page_title="Tweet Dashboard", layout="wide")

# ----------------------------------------------
# Helper Functions
# ----------------------------------------------
@st.cache_data
def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

analyzer = SentimentIntensityAnalyzer()

def sentiment_label(text):
    s = analyzer.polarity_scores(str(text))["compound"]
    if s >= 0.05:
        return "Positive"
    elif s <= -0.05:
        return "Negative"
    return "Neutral"

def make_wordcloud(text):
    if not text:
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    return wc

# ----------------------------------------------
# Load Data
# ----------------------------------------------
tweets = load_csv_safe("data/twitter_tweets_clean.csv")
clusters = load_csv_safe("data/twitter_clusters.csv")
sentiment_df = load_csv_safe("data/twitter_sentiment.csv")

# ----------------------------------------------
# Merge Data Safely
# ----------------------------------------------
if not tweets.empty and not clusters.empty:
    # merge by index only (your old working method)
    merged = pd.concat([tweets.reset_index(drop=True), clusters["cluster"].reset_index(drop=True)], axis=1)
else:
    merged = tweets.copy()
    merged["cluster"] = 0

# Apply sentiment if missing
if not sentiment_df.empty:
    merged["sentiment"] = sentiment_df.get("sentiment", pd.Series(dtype=str))
else:
    if "sentiment" not in merged.columns:
        merged["sentiment"] = merged["clean_text"].apply(sentiment_label)

# ----------------------------------------------
# Sidebar Filters
# ----------------------------------------------
st.sidebar.title("Dashboard Filters")

# Clusters
cluster_options = sorted(merged["cluster"].unique().tolist())
selected_cluster = st.sidebar.multiselect("Cluster(s)", options=cluster_options, default=cluster_options)

# Sentiments
sentiment_options = ["Positive", "Neutral", "Negative"]
selected_sentiment = st.sidebar.multiselect("Sentiment", sentiment_options, sentiment_options)

# Search
search_text = st.sidebar.text_input("Search in tweets")

# Date Filter (NO TIMEZONE ERRORS)
start_date = end_date = None
if "created_at" in merged.columns:
    merged["created_at"] = pd.to_datetime(merged["created_at"], errors="coerce")
    merged["created_at"] = merged["created_at"].dt.tz_localize(None)  # remove timezone
    merged = merged.dropna(subset=["created_at"])

    if not merged.empty:
        min_d = merged["created_at"].min().date()
        max_d = merged["created_at"].max().date()

        start_date, end_date = st.sidebar.date_input(
            "Date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d
        )

# ----------------------------------------------
# Apply Filters
# ----------------------------------------------
df_display = merged.copy()

if selected_cluster:
    df_display = df_display[df_display["cluster"].isin(selected_cluster)]

if selected_sentiment:
    df_display = df_display[df_display["sentiment"].isin(selected_sentiment)]

if search_text:
    df_display = df_display[df_display["clean_text"].str.contains(search_text, case=False, na=False)]

if start_date and end_date:
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    df_display = df_display[(df_display["created_at"] >= sd) & (df_display["created_at"] < ed)]

# ----------------------------------------------
# Page Header
# ----------------------------------------------
st.title("Tweet Analysis Dashboard")
st.write("Interactive sentiment and cluster-based analysis on Twitter data.")

# ----------------------------------------------
# Metrics
# ----------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Tweets", len(df_display))
col2.metric("Clusters", df_display["cluster"].nunique())
col3.metric("Positive %", f"{(df_display['sentiment']=='Positive').mean()*100:.1f}%")

# ----------------------------------------------
# Sentiment Chart
# ----------------------------------------------
st.header("Sentiment Distribution")

fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=df_display, x="sentiment", order=["Positive", "Neutral", "Negative"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# ----------------------------------------------
# Cluster-wise Sentiment
# ----------------------------------------------
st.header("Cluster-wise Sentiment Distribution")

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.countplot(data=df_display, x="cluster", hue="sentiment", palette="coolwarm", ax=ax2)
st.pyplot(fig2)

# ----------------------------------------------
# Wordcloud
# ----------------------------------------------
st.header("Cluster WordClouds")

if selected_cluster:
    for c in selected_cluster:
        st.subheader(f"Cluster {c}")
        texts = df_display[df_display["cluster"] == c]["clean_text"].dropna().astype(str).tolist()
        wc = make_wordcloud(texts)

        if wc:
            fig_wc, ax_wc = plt.subplots(figsize=(8,4))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.write("No text available.")

# ----------------------------------------------
# Data Table
# ----------------------------------------------
st.header("Tweets (Filtered)")
st.dataframe(df_display[['clean_text', 'sentiment', 'cluster']].head(50))

# ----------------------------------------------
# Export Button
# ----------------------------------------------
st.download_button(
    "Download CSV",
    df_display.to_csv(index=False).encode("utf-8"),
    "filtered_tweets.csv",
    "text/csv"
)
