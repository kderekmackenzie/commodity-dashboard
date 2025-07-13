from pinecone_setup import build_pinecone_index 
import os

# Try to load from Streamlit secrets if available (production)
try:
    import streamlit as st
    api_key = st.secrets.get("OPENAI_API_KEY")
except ModuleNotFoundError:
    api_key = None

# Fallback to .env for local use
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Add it to .env or Streamlit secrets.")

build_pinecone_index()
