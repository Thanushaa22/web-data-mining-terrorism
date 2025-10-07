# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import os

st.set_page_config(page_title="Web Data Mining - Tweet Analysis", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

analyzer = SentimentIntensityAnalyzer()

def sentiment_label(text):
    s = analyzer.polarity_scores(str(text))['compound']
    return "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")

@st.cache_data(ttl=300)
def fetch_live_tweets(query, max_results, bearer_token):
    # Import local collector (ensure path is correct)
    from src.collect.twitter_collector import collect_tweets
    df = collect_tweets(query=query, max_results=max_results, bearer_token=bearer_token)
    return df

def make_wordcloud(text, width=800, height=400):
    if not text:
        return None
    wc = WordCloud(width=width, height=height, background_color='white').generate(" ".join(text))
    return wc

# ---------- Load stored data ----------
tweets = load_csv_safe("data/twitter_tweets_clean.csv")
features = load_csv_safe("data/twitter_features.csv")
clusters = load_csv_safe("data/twitter_clusters.csv")
sentiment_df = load_csv_safe("data/twitter_sentiment.csv")

# Merge for UI (safe merges even if some files are missing)
if not tweets.empty and not clusters.empty:
    merged = pd.concat([tweets.reset_index(drop=True), clusters["cluster"].reset_index(drop=True)], axis=1)
else:
    merged = tweets.copy()
    if not merged.empty:
        merged['cluster'] = 0

if not sentiment_df.empty:
    merged['sentiment'] = sentiment_df.get('sentiment', pd.Series(dtype=str))
else:
    if 'sentiment' not in merged.columns:
        merged['sentiment'] = merged.get('clean_text', '').apply(sentiment_label) if not merged.empty else pd.Series(dtype=str)

# ---------- Sidebar (filters and live fetch) ----------
st.sidebar.title("Filters & Live Fetch")
cluster_options = sorted(merged['cluster'].dropna().unique().tolist()) if 'cluster' in merged else []
selected_cluster = st.sidebar.multiselect("Cluster(s)", options=cluster_options, default=cluster_options)

sentiment_options = ["Positive", "Neutral", "Negative"]
selected_sentiment = st.sidebar.multiselect("Sentiment", options=sentiment_options, default=sentiment_options)

search_text = st.sidebar.text_input("Search text (free text)")

# Date filter (if you have created_at column)
if 'created_at' in merged:
    merged['created_at'] = pd.to_datetime(merged['created_at'], errors='coerce')
    start_date, end_date = st.sidebar.date_input("Date range", value=(None, None))
else:
    start_date = end_date = None

st.sidebar.markdown("---")
st.sidebar.subheader("Live tweet fetch")
query = st.sidebar.text_input("Query", value="terrorism OR extremist")
max_results = st.sidebar.slider("Max tweets", 10, 100, 50)
use_live = st.sidebar.button("Fetch live tweets")

# ---------- Main layout ----------
st.title("🛰 Web Data Mining: Twitter Terrorism Analysis")
st.markdown("Analyze tweet clusters, keywords, and sentiment.")

# Overview
col1, col2, col3 = st.columns(3)
col1.metric("Total Tweets", len(merged))
col2.metric("Total Clusters", merged['cluster'].nunique() if 'cluster' in merged else 0)
col3.metric("Positive %", f"{(merged['sentiment']=='Positive').mean()*100:.1f}%" if not merged.empty else "0%")

# Apply filters
df_display = merged.copy()
if selected_cluster:
    df_display = df_display[df_display['cluster'].isin(selected_cluster)]
if selected_sentiment:
    df_display = df_display[df_display['sentiment'].isin(selected_sentiment)]
if search_text:
    df_display = df_display[df_display['clean_text'].str.contains(search_text, case=False, na=False)]
if start_date and end_date and 'created_at' in df_display:
    df_display = df_display[(df_display['created_at'] >= pd.to_datetime(start_date)) & (df_display['created_at'] <= pd.to_datetime(end_date))]

# Download button for the filtered dataframe
csv_bytes = df_display.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download filtered tweets (CSV)", csv_bytes, file_name="filtered_tweets.csv", mime="text/csv")

# Sentiment distribution
st.header("Sentiment Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="sentiment", data=df_display, order=sentiment_options, palette="coolwarm", ax=ax)
st.pyplot(fig)

# Cluster-wise sentiment
if 'cluster' in df_display:
    st.header("Cluster-wise Sentiment Distribution")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.countplot(x="cluster", hue="sentiment", data=df_display, palette="coolwarm", ax=ax2)
    st.pyplot(fig2)

# Word cloud for selected cluster
if selected_cluster:
    for c in selected_cluster:
        st.subheader(f"WordCloud — Cluster {c}")
        texts = df_display[df_display['cluster']==c]['clean_text'].dropna().astype(str).tolist()
        wc = make_wordcloud(texts)
        if wc:
            fig_wc, ax_wc = plt.subplots(figsize=(8,4))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.write("No text to generate wordcloud")

# Table preview
st.header("🔍 Sample Tweets (filtered)")
st.dataframe(df_display[['clean_text','sentiment','cluster']].head(50))

# ---------- Live fetch behavior ----------
if use_live:
    # get token from streamlit secrets first
    bearer = None
    if st.secrets and "TWITTER_BEARER_TOKEN" in st.secrets:
        bearer = st.secrets["AAAAAAAAAAAAAAAAAAAAAJ874gEAAAAA118IZbotZ5gCsUA2G9KIDKHqKqw%3DkZ1pSZxog2yXQQPr3Kw5JZvdiLAChPsgCgxAu4P549ZExOGS5c"]
    else:
        bearer = os.getenv("AAAAAAAAAAAAAAAAAAAAAJ874gEAAAAA118IZbotZ5gCsUA2G9KIDKHqKqw%3DkZ1pSZxog2yXQQPr3Kw5JZvdiLAChPsgCgxAu4P549ZExOGS5c")
    if not bearer:
        st.error("TWITTER_BEARER_TOKEN is missing. Add it to Streamlit Secrets or environment variables.")
    else:
        with st.spinner("Fetching tweets..."):
            try:
                new_df = fetch_live_tweets(query=query, max_results=max_results, bearer_token=bearer)
                if new_df.empty:
                    st.warning("No tweets returned for that query.")
                else:
                    # compute clean_text and sentiment for live fetch
                    # reuse simple cleaning here (or call your preprocess functions)
                    new_df['clean_text'] = new_df['text'].fillna("").str.replace(r"http\S+|www\S+|@\w+|#\w+", "", regex=True).str.replace(r"[^A-Za-z\s]","",regex=True).str.lower()
                    new_df['sentiment'] = new_df['clean_text'].apply(sentiment_label)
                    st.success(f"Fetched {len(new_df)} tweets")
                    st.dataframe(new_df[['clean_text','sentiment','retweets','likes']].head(50))
                    # allow download of fetched data
                    st.download_button("Download fetched tweets (CSV)", new_df.to_csv(index=False).encode('utf-8'), file_name="fetched_tweets.csv")
            except Exception as e:
                st.error(f"Fetch failed: {e}")
