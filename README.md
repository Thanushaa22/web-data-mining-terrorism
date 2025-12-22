🛰 Web Data Mining – Terrorism Tweet Analysis

A Streamlit-based web application that analyzes terrorism-related tweets using web data mining and NLP techniques.
The project focuses on tweet preprocessing, sentiment analysis, clustering, and visual insights to understand online discussions around extremism.

📌 Project Objective

To demonstrate how web mining and data analytics can be used to:

-Analyze social media data related to terrorism
-Identify sentiment patterns and clusters
-Provide meaningful visual insights for decision-making

🚀 Features

📊 Dashboard Analytics
1.Tweet count, sentiment distribution
2.Cluster-wise analysis
3.Date-wise trends

🧠 Text Processing
1.Tweet cleaning (URLs, mentions, hashtags removal)
2.Token normalization
3.Keyword extraction

😊 Sentiment Analysis
1.Positive / Neutral / Negative classification
2.Implemented using VADER Sentiment Analyzer

☁ WordCloud Visualization
1.Displays dominant keywords from filtered tweets

⚡ Live Tweet Fetch (API-based)
1.Uses Twitter API (Bearer Token required)
2.Limited in deployment due to API rate restrictions

📥 CSV Export
1.Download filtered and processed tweet data

🛠 Technologies Used

-Python
-Streamlit – Web UI
-Pandas & NumPy – Data processing
-Matplotlib / Seaborn / Plotly – Visualization
-VADER Sentiment Analyzer – NLP sentiment analysis
-Twitter API (Tweepy) – Live data collection

📂 Project Structure
web-data-mining-terrorism/
│
├── app.py                     # Main Streamlit application
├── data/
│   ├── twitter_tweets_clean.csv
│   ├── twitter_clusters.csv
│   ├── twitter_sentiment.csv
│
├── src/
│   └── collect/
│       └── twitter_collector.py
│
├── requirements.txt
└── README.md

▶ How to Run Locally

1.Clone the repository
git clone https://github.com/Thanushaa22/web-data-mining-terrorism.git
cd web-data-mining-terrorism

2.Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3.Install dependencies
pip install -r requirements.txt

4.Run the app
streamlit run app.py

🔐 Twitter API Configuration (Optional)
To enable live tweet fetching:
-Create a Twitter Developer account
-Generate a Bearer Token
-Add it to Streamlit secrets:
(.streamlit/secrets.toml)
TWITTER_BEARER_TOKEN = "your_token_here"
⚠️ Note: Live API may be limited on deployment due to rate limits.

🌐 Deployment
-Deployed using Streamlit Cloud
-Static analysis features work reliably
-Live tweet fetching may not work consistently due to API restrictions

---
Demo link:
https://web-data-mining-terrorism-k3ytqpgwpj3oshvtrvan4s.streamlit.app/







