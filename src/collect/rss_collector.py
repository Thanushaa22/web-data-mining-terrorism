import feedparser
import pandas as pd
from datetime import datetime
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml"
}

def collect_rss_articles():
    articles = []
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            articles.append({
                "source": source,
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", datetime.utcnow().isoformat()),
                "summary": entry.get("summary", "")
            })
    return pd.DataFrame(articles)

if __name__ == "__main__":
    df = collect_rss_articles()
    if not df.empty:
        df.to_csv("data/rss_articles.csv", index=False, encoding="utf-8")
        print(f"✅ Collected {len(df)} articles and saved to data/rss_articles.csv")
        print(df.head())
    else:
        print("⚠️ No articles collected")
