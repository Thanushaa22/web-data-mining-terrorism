# src/collect/twitter_collector.py
import os
import time
import tweepy
import pandas as pd
from datetime import datetime

def get_twitter_client(bearer_token=None):
    token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        raise ValueError("Twitter bearer token not provided. Set TWITTER_BEARER_TOKEN env or pass bearer_token.")
    client = tweepy.Client(bearer_token=token, wait_on_rate_limit=False)
    return client

def collect_tweets(query="terrorism OR extremist", max_results=50, bearer_token=None):
    client = get_twitter_client(bearer_token=bearer_token)
    q = f"{query} -is:retweet lang:en"
    all_tweets = []
    # max_results up to 100 per request; do small pages
    remaining = max_results
    next_token = None
    backoff = 1
    while remaining > 0:
        count = min(100, remaining)
        try:
            resp = client.search_recent_tweets(query=q,
                                              max_results=count,
                                              next_token=next_token,
                                              tweet_fields=["id","text","created_at","public_metrics","author_id"])
        except tweepy.TooManyRequests:
            # exponential backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        except Exception as e:
            raise e

        if not resp or not resp.data:
            break
        for t in resp.data:
            pm = t.public_metrics if hasattr(t, "public_metrics") else {}
            all_tweets.append({
                "id": t.id,
                "author_id": getattr(t, "author_id", None),
                "text": t.text,
                "created_at": t.created_at,
                "retweets": pm.get("retweet_count", 0),
                "likes": pm.get("like_count", 0),
                "collected_at": datetime.utcnow().isoformat()
            })
        remaining -= len(resp.data)
        # pagination token
        next_token = getattr(resp.meta, "next_token", None) or resp.meta.get("next_token") if resp.meta else None
        if not next_token:
            break
    return pd.DataFrame(all_tweets)
