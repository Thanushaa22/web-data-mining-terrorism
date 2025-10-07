import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Web Data Mining - Tweet Analysis", layout="wide")

st.title("🛰 Web Data Mining: Twitter Terrorism Analysis")
st.write("Analyze tweet clusters, top keywords, and sentiment trends in real-time.")

# Load data
tweets = pd.read_csv("data/twitter_tweets_clean.csv")
features = pd.read_csv("data/twitter_features.csv")
clusters = pd.read_csv("data/twitter_clusters.csv")
sentiment = pd.read_csv("data/twitter_sentiment.csv")

# Merge data
merged = pd.concat([tweets, clusters["cluster"], sentiment["sentiment"]], axis=1)

# --- Section 1: Summary ---
st.header("📊 Overview")
st.write(f"Total Tweets: **{len(merged)}**")
st.write(f"Total Clusters: **{merged['cluster'].nunique()}**")

# --- Section 2: Sentiment Distribution ---
st.header("💬 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="sentiment", data=merged, palette="coolwarm", ax=ax)
st.pyplot(fig)

# --- Section 3: Cluster-wise Sentiment ---
st.header("🎯 Cluster-wise Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x="cluster", hue="sentiment", data=merged, palette="coolwarm", ax=ax)
st.pyplot(fig)

# --- Section 4: Explore Individual Tweets ---
st.header("🔍 Explore Tweets by Cluster")
selected_cluster = st.selectbox("Select Cluster", merged["cluster"].unique())
filtered = merged[merged["cluster"] == selected_cluster]
st.dataframe(filtered[["clean_text", "sentiment"]].head(20))

st.success("✅ Dashboard Loaded Successfully!")
