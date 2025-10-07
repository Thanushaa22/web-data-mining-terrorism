import pandas as pd
import numpy as np

# Load clustered tweets and features
features_df = pd.read_csv("data/twitter_features.csv")
clusters_df = pd.read_csv("data/twitter_clusters.csv")

# Merge both DataFrames to align cluster labels
merged = clusters_df.copy()
merged["cluster"] = clusters_df["cluster"]

# Drop 'id' and keep only text features + cluster
feature_cols = [c for c in features_df.columns if c != "id"]
merged_features = merged[feature_cols + ["cluster"]]

# Calculate average TF-IDF weight per feature for each cluster
top_words = {}
for c in sorted(merged["cluster"].unique()):
    cluster_data = merged_features[merged_features["cluster"] == c]
    mean_tfidf = cluster_data[feature_cols].mean().sort_values(ascending=False)
    top_words[c] = mean_tfidf.head(10).index.tolist()

# Print top 10 words for each cluster
print("\n🔍 Top Keywords per Cluster:\n")
for cluster, words in top_words.items():
    print(f"Cluster {cluster}: {', '.join(words)}")
