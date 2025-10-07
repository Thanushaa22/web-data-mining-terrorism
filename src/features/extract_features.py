import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load cleaned tweets
df = pd.read_csv("data/twitter_tweets_clean.csv")

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=2500, stop_words="english")

# Fit and transform the clean_text column
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.insert(0, "id", df.index)

# Save features
output_path = "data/twitter_features.csv"
tfidf_df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Extracted TF-IDF features: {tfidf_df.shape[0]} tweets × {tfidf_df.shape[1]-1} features")
print(tfidf_df.head())
