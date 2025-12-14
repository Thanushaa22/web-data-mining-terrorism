import streamlit as st
import pandas as pd
import os
from datetime import datetime
import tweepy

st.set_page_config(page_title="Live Tweet Monitor", layout="wide")

# ================= GET API CLIENT =================
def get_client():
    token = st.secrets.get("TWITTER_BEARER_TOKEN", os.getenv("TWITTER_BEARER_TOKEN"))
    if not token:
        st.error("Twitter API token missing.")
        return None
    return tweepy.Client(bearer_token=token, wait_on_rate_limit=False)

# ================= FETCH LIVE TWEETS =================
@st.cache_data(ttl=30)    # prevents rate-limit from repeated clicks
def fetch(query, limit):
    client = get_client()
    if not client:
        return pd.DataFrame(), "TOKEN"

    raw_q = f"{query} lang:en -is:retweet"
    try:
        res = client.search_recent_tweets(
            query=raw_q,
            max_results=limit,
            tweet_fields=["id", "author_id", "text", "created_at", "public_metrics"]
        )
    except tweepy.TooManyRequests:
        return pd.DataFrame(), "RATE"
    except Exception as e:
        return pd.DataFrame(), f"ERROR {e}"

    if not res.data:
        return pd.DataFrame(), "EMPTY"

    data = []
    for t in res.data:
        m = t.public_metrics
        data.append({
            "created_at": t.created_at,
            "text": t.text,
            "retweets": m["retweet_count"],
            "likes": m["like_count"],
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return pd.DataFrame(data), "OK"

# ================= UI =================
st.title("Live Tweet Monitor")
st.write("Fetch real-time tweets related to terrorism and extremist activities.")

query = st.text_input("Enter search keywords", "terrorism OR extremist")
limit = st.slider("Number of tweets", 10, 50, 20)
go = st.button("Fetch Tweets")

if go:
    with st.spinner("Fetching..."):
        df, status = fetch(query, limit)

    if status == "RATE":
        st.warning("API limit reached. Try again in 20–30 seconds.")
    elif status == "TOKEN":
        st.error("Token missing.")
    elif status == "EMPTY":
        st.info("No tweets found for this query.")
    elif status.startswith("ERROR"):
        st.error(status)
    else:
        st.success(f"Fetched {len(df)} tweets!")
        st.dataframe(df)
        st.download_button("⬇ Download CSV", df.to_csv(index=False), "live_tweets.csv")
else:
    st.info("Enter query and press **Fetch Tweets** to begin.")
