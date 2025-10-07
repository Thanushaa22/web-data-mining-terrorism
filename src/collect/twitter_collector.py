import tweepy
import pandas as pd
import os
from datetime import datetime

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# 🔑 Replace with your own Bearer Token from Twitter Developer Portal
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJ874gEAAAAA118IZbotZ5gCsUA2G9KIDKHqKqw%3DkZ1pSZxog2yXQQPr3Kw5JZvdiLAChPsgCgxAu4P549ZExOGS5c"

# Initialize Tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def collect_tweets(query="terrorism OR extremist", max_results=50):
    """Collect tweets based on a search query"""
    tweets = []
    response = client.search_recent_tweets(
        query=query + " lang:en -is:retweet",
        tweet_fields=["id", "text", "author_id", "created_at", "public_metrics"],
        max_results=max_results
    )
    if response.data:
        for tweet in response.data:
            tweets.append({
                "id": tweet.id,
                "author_id": tweet.author_id,
                "created_at": tweet.created_at,
                "text": tweet.text,
                "retweets": tweet.public_metrics.get("retweet_count", 0),
                "likes": tweet.public_metrics.get("like_count", 0),
                "query": query,
                "collected_at": datetime.utcnow().isoformat()
            })
    return pd.DataFrame(tweets)

if __name__ == "__main__":
    df = collect_tweets()
    if not df.empty:
        df.to_csv("data/twitter_tweets.csv", index=False, encoding="utf-8")
        print(f"✅ Collected {len(df)} tweets → saved to data/twitter_tweets.csv")
        print(df.head())
    else:
        print("⚠️ No tweets collected. Check your query or API access.")
