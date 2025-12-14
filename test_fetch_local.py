# test_fetch_local.py
import os, traceback
from src.collect.twitter_collector import collect_tweets

print("PWD:", __import__('os').getcwd())
token = None
# prefer Streamlit secrets file path check (not used here) — just env
token = os.getenv("AAAAAAAAAAAAAAAAAAAAAJ874gEAAAAA118IZbotZ5gCsUA2G9KIDKHqKqw%3DkZ1pSZxog2yXQQPr3Kw5JZvdiLAChPsgCgxAu4P549ZExOGS5c")
print("ENV token present:", bool(token))

try:
    df = collect_tweets(query="terrorism OR extremist", max_results=10, bearer_token=token)
    print("Type:", type(df))
    if df is None:
        print("collect_tweets returned None")
    else:
        print("Rows returned:", len(df))
        print(df.head(5).to_dict(orient='records'))
except Exception as e:
    print("ERROR in collect_tweets:")
    traceback.print_exc()
