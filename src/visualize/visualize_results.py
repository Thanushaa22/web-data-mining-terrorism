import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load sentiment and cluster data
sentiment_df = pd.read_csv("data/twitter_sentiment.csv")
clusters_df = pd.read_csv("data/twitter_clusters.csv")

# Merge both
merged = pd.concat([sentiment_df, clusters_df["cluster"]], axis=1)

# --- 1️⃣ Overall Sentiment Distribution ---
plt.figure(figsize=(7, 5))
sns.countplot(x="sentiment", data=merged, palette="coolwarm")
plt.title("Overall Tweet Sentiment Distribution")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Tweets")
plt.show()

# --- 2️⃣ Sentiment per Cluster ---
plt.figure(figsize=(10, 6))
sns.countplot(x="cluster", hue="sentiment", data=merged, palette="coolwarm")
plt.title("Sentiment Distribution per Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Tweets")
plt.legend(title="Sentiment")
plt.show()
