import pandas as pd
from sklearn.cluster import KMeans
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

def cluster_articles():
    # Load features
    df_features = pd.read_csv("data/rss_features.csv")
    
    # Drop the ID column
    X = df_features.drop(columns=["id"])
    
    # Run KMeans clustering (let’s try 5 clusters)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_features["cluster"] = kmeans.fit_predict(X)
    
    # Save clustered data
    df_features.to_csv("data/rss_clusters.csv", index=False)
    print(f"✅ Clustered {len(df_features)} articles into 5 groups")
    print(df_features[["id", "cluster"]].head())

if __name__ == "__main__":
    cluster_articles()
