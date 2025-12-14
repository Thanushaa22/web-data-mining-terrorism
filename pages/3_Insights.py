import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

st.set_page_config(page_title="Tweet Insights", layout="wide")

# ---------------- Load CSV safely ----------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

tweets = load_csv("data/twitter_tweets_clean.csv")
clusters = load_csv("data/twitter_clusters.csv")
sentiments = load_csv("data/twitter_sentiment.csv")

# Merge Data
if not tweets.empty and not clusters.empty:
    df = pd.concat([tweets.reset_index(drop=True), clusters["cluster"].reset_index(drop=True)], axis=1)
else:
    df = tweets.copy()
    if not df.empty:
        df['cluster'] = 0

if not sentiments.empty:
    df['sentiment'] = sentiments['sentiment']
else:
    df['sentiment'] = "Neutral"

# ---------------- Header UI ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0F2027,#203A43,#2C5364);
    color: white !important;
}
.header {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #04d9ff;
    margin-bottom: -15px;
}
.sub {
    text-align: center;
    font-size: 19px;
    color: #d6f6ff;
    margin-bottom: 35px;
}
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 25px;
    color: #00eaff;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    margin: 8px 0;
    border-radius: 12px;
    backdrop-filter: blur(6px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>Deep Tweet Insight Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Explore trending topics, strongest emotions & cluster patterns</div>", unsafe_allow_html=True)

# ---------------- Key Metrics ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Tweets", len(df))
c2.metric("Clusters Found", df['cluster'].nunique())
c3.metric("Dominant Sentiment", df['sentiment'].mode()[0] if not df.empty else "N/A")

# ---------------- Insight 1: Sentiment vs Clusters ----------------
st.markdown("<div class='section-title'>Sentiment Strength in Each Cluster</div>", unsafe_allow_html=True)
if not df.empty:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x="cluster", hue="sentiment", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Dataset is empty — upload data first.")

# ---------------- Insight 2: Top Words per Cluster ----------------
st.markdown("<div class='section-title'>Most Frequent Words in Each Cluster</div>", unsafe_allow_html=True)
cluster_id = st.selectbox("Select Cluster", sorted(df['cluster'].unique().tolist()))

words = " ".join(df[df["cluster"] == cluster_id]["clean_text"].dropna())
if words.strip():
    wc = WordCloud(background_color="black", colormap="cool", width=800, height=400).generate(words)
    fig_wc, ax_wc = plt.subplots(figsize=(8,4))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)
else:
    st.info("No words to visualize in this cluster.")

# ---------------- Insight 3: Top 15 Most Discussed Terms ----------------
st.markdown("<div class='section-title'>Top 15 Most Repeated Words</div>", unsafe_allow_html=True)

if not df.empty:
    text = " ".join(df["clean_text"].dropna())
    words = pd.Series(text.split()).value_counts().head(15)
    st.bar_chart(words)
else:
    st.warning("No cleaned tweets found.")

# ---------------- Insight 4: Sentiment Trend Over Time ----------------
st.markdown("<div class='section-title'>Sentiment Trend Over Time</div>", unsafe_allow_html=True)

if "created_at" in df.columns:
    try:
        trend = df.copy()
        trend["created_at"] = pd.to_datetime(trend["created_at"], errors="coerce")
        trend["day"] = trend["created_at"].dt.date
        counts = trend.groupby(["day","sentiment"]).size().unstack(fill_value=0)
        st.line_chart(counts)
    except:
        st.warning("Trend graph skipped due to date format issue.")
else:
    st.info("No timestamp data available.")

# ---------------- Raw Data ----------------
st.markdown("<div class='section-title'>Raw Data Explorer</div>", unsafe_allow_html=True)
st.dataframe(df[['clean_text','sentiment','cluster']].head(50))

# ---------------- Footer ----------------
st.markdown("""
<hr><p style='text-align:center;color:#9ed9ff'>
</p>
""", unsafe_allow_html=True)
