import pandas as pd
import numpy as np

def summarize_clusters():
    # Load features + clusters
    df = pd.read_csv("data/rss_clusters.csv")

    # Drop ID & cluster columns to isolate features
    feature_cols = [col for col in df.columns if col not in ["id", "cluster"]]
    
    # Calculate average TF-IDF score per cluster
    cluster_summary = {}
    for cluster in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster][feature_cols]
        mean_scores = cluster_data.mean().sort_values(ascending=False)
        top_keywords = mean_scores.head(10).index.tolist()
        cluster_summary[cluster] = top_keywords

    # Print summaries
    for cluster, keywords in cluster_summary.items():
        print(f"\n🟢 Cluster {cluster} top keywords:")
        print(", ".join(keywords))

if __name__ == "__main__":
    summarize_clusters()
