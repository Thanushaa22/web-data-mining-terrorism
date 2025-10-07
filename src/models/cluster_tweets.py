import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Load TF-IDF features
df = pd.read_csv("data/twitter_features.csv")

# Drop 'id' column for clustering
X = df.drop(columns=["id"])

# Choose number of clusters (you can adjust this)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Save clustered data
os.makedirs("data", exist_ok=True)
df.to_csv("data/twitter_clusters.csv", index=False)
print(f"✅ Clustered {len(df)} tweets into {n_clusters} groups → saved to data/twitter_clusters.csv")

# Optional: reduce to 2D for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=df["cluster"], cmap="tab10")
plt.title("Tweet Clusters (K-Means)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
