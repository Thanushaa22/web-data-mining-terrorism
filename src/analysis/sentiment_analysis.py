import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load cleaned tweets
df = pd.read_csv("data/twitter_tweets_clean.csv")

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Return overall sentiment label for given text"""
    score = analyzer.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["sentiment"] = df["clean_text"].apply(get_sentiment)

# Save results
output_path = "data/twitter_sentiment.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Analyzed sentiment for {len(df)} tweets → saved to {output_path}")
print(df[["clean_text", "sentiment"]].head(10))
