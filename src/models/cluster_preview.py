import pandas as pd

def preview_clusters():
    # Load clustered features
    df_features = pd.read_csv("data/rss_clusters.csv")
    # Load original articles
    df_articles = pd.read_csv("data/rss_articles.csv")

    # Merge so we have titles + clusters
    df = df_features[["id", "cluster"]].merge(
        df_articles[["title", "summary"]], left_on="id", right_index=True
    )

    # Show 3 sample articles per cluster
    for cluster in sorted(df["cluster"].unique()):
        print(f"\n🟢 Cluster {cluster} sample articles:")
        cluster_samples = df[df["cluster"] == cluster].head(3)
        for _, row in cluster_samples.iterrows():
            print(f" - {row['title']}")

if __name__ == "__main__":
    preview_clusters()
