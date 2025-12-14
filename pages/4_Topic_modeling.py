import streamlit as st
import pandas as pd
import re
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px

st.set_page_config(page_title="Topic Modeling", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    path = "data/twitter_tweets_clean.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.dropna(subset=["clean_text"])
        return df
    else:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No cleaned tweet data found! Run preprocessing first.")
    st.stop()

# ---------- UI ----------
st.title("Topic Modeling & Theme Discovery")
st.write("Uncover hidden themes in terrorism-related tweets using LDA topic modeling.")

num_topics = st.slider("Select Number of Topics", 2, 10, 5)

# ---------- Preprocess Text ----------
texts = df["clean_text"].astype(str).tolist()

vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# ---------- LDA Model ----------
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

topic_keywords = []
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    topic_keywords.append(", ".join(reversed(top_words)))

# ---------- Visualization: Topic Bar Chart ----------
topic_df = pd.DataFrame({"Topic": [f"Topic {i+1}" for i in range(num_topics)],
                        "Keywords": topic_keywords})

fig = px.bar(topic_df,
             x="Topic",
             y=[sum(x) for x in lda.components_],
             text="Keywords",
             title="Topic Importance",
             color="Topic")
fig.update_traces(textposition="inside")
st.plotly_chart(fig, use_container_width=True)

# ---------- WordCloud for each topic ----------
st.header("Topic WordClouds")
for i in range(num_topics):
    st.subheader(f"Topic {i+1} Keywords")
    text = topic_keywords[i].replace(",", "")
    wc = WordCloud(width=700, height=350, background_color="black", colormap="cool")
    img = wc.generate(text)
    st.image(img.to_array())

# ---------- Show Keyword Table ----------
st.subheader("Topic Keywords Table")
st.dataframe(topic_df)

# ---------- Footer ----------
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
</p>
""", unsafe_allow_html=True)
