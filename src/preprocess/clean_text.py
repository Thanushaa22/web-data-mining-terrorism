import pandas as pd
import re
import spacy
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_tweet_text(text):
    """Clean tweet text by removing mentions, hashtags, links, etc."""
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                    # remove mentions
    text = re.sub(r"#\w+", "", text)                    # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)             # remove punctuation/emojis
    text = text.lower().strip()                         # lowercase and trim
    return text

def lemmatize(text):
    """Lemmatize and remove stopwords"""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

if __name__ == "__main__":
    # Load tweets
    df = pd.read_csv("data/twitter_tweets.csv")

    # Apply cleaning + lemmatization
    df["clean_text"] = df["text"].fillna("").apply(clean_tweet_text).apply(lemmatize)

    # Save cleaned data
    df.to_csv("data/twitter_tweets_clean.csv", index=False, encoding="utf-8")
    print(f"✅ Cleaned {len(df)} tweets → saved to data/twitter_tweets_clean.csv")
    print(df[["text", "clean_text"]].head())
