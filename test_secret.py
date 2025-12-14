import streamlit as st, os

st.write("Current Working Directory:", os.getcwd())
st.write("Secret paths Streamlit checks:")
st.write("1.", os.path.expanduser("~/.streamlit/secrets.toml"))
st.write("2.", os.path.join(os.getcwd(), ".streamlit", "secrets.toml"))

st.write("Found:", "TWITTER_BEARER_TOKEN" in st.secrets)
st.write("Token Value:", st.secrets.get("TWITTER_BEARER_TOKEN", None))
