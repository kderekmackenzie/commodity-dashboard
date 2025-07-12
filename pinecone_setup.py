import pandas as pd
import os
from dotenv import load_dotenv 
from openai import OpenAI
from pinecone import Pinecone

from typing import List, Dict

# --- Load environment variables immediately ---
load_dotenv() # Load .env for local development

# --- Constants ---
DATA_PATH = "commodity-dashboard-prices.csv"
INDEX_NAME = "commodity-dash" # Match the index name you created in Pinecone

# --- Initialize API Keys & Environment ---
# Access environment variables directly from os.environ
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print(f"DEBUG (Direct Load): PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"DEBUG (Direct Load): PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"DEBUG (Direct Load): OPENAI_API_KEY: {OPENAI_API_KEY}")

# This check should now pass if .env is correct
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY]):
    raise ValueError("❌ Pinecone or OpenAI API keys/environment not found. Add them to .env.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# --- OpenAI Embedding Function ---
class OpenAIEmbeddingFunction:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=input, model="text-embedding-ada-002")
        return [data.embedding for data in response.data]

embed_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)


# --- Data Preprocessing ---
def preprocess_data(file_path: str) -> tuple[list[str], list[str], list[dict]]:
    print(f"📥 Reading file: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'], na_values=["...", "…"])
    print("✅ CSV loaded:", df.shape)

    df.set_index('Date', inplace=True)

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("…", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.resample('YE').mean().dropna(how='all')
    df['Year'] = df.index.year

    print("📊 After resampling:", df.shape)

    docs, ids, metas = [], [], []

    for col in df.columns.drop('Year'):
        for decade_start in range(1960, 2030, 10):
            decade_df = df[(df['Year'] >= decade_start) & (df['Year'] < decade_start + 10)]
            if decade_df.empty or decade_df[col].isna().all():
                continue

            decade_str = f"{decade_start}s"
            text_content = f"{col} prices in the {decade_str}: " + \
                            ", ".join([f"{row['Year']}: {row[col]:.2f}" for _, row in decade_df.iterrows() if pd.notna(row[col])])
            
            doc_id = f"{col}-{decade_str}"
            docs.append(text_content) # This list will be embedded
            ids.append(doc_id)
            metas.append({
                "commodity": col,
                "decade": decade_str,
                "type": "price_data",
                "text_content": text_content # Add the full text content to metadata for retrieval
            })

    print(f"✅ Preprocessing complete: {len(docs)} chunks prepared.")
    return docs, ids, metas


# --- Pinecone-specific functions ---
def build_pinecone_index():
    """
    Builds or connects to the Pinecone index and uploads data.
    """
    print(f"📥 Initializing Pinecone Index '{INDEX_NAME}'...")
    
    # DEBUG: Checking Pinecone index existence now (expected line 100 or near it).
    # Corrected line for Pinecone client 7.x.x:
    # Iterate through the index objects to get their names
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]: 
        print(f"🚀 Creating new Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536, # OpenAI embedding dimension
            metric='cosine',
            cloud="aws", 
            region="us-east-1"
        )
        print(f"✅ Index '{INDEX_NAME}' created.")
    else:
        print(f"📂 Connecting to existing Pinecone index '{INDEX_NAME}'.")

    index = pc.Index(INDEX_NAME)

    # Preprocess data and prepare for upsert
    docs, ids, metas = preprocess_data(DATA_PATH)
    print(f"📝 {len(docs)} documents prepared for insertion")

    # Check if index is empty before upserting
    if index.describe_index_stats().total_vector_count == 0:
        print(f"Attempting to add {len(docs)} documents...")
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = metas[i : i + batch_size]

            batch_embeddings = embed_fn(batch_docs)

            vectors_to_upsert = []
            for j in range(len(batch_docs)):
                vectors_to_upsert.append({
                    "id": batch_ids[j],
                    "values": batch_embeddings[j],
                    "metadata": batch_metas[j]
                })
            
            index.upsert(vectors=vectors_to_upsert)
        print(f"✅ Added {len(docs)} documents to Pinecone index.")
    else:
        print(f"⚠️ Index '{INDEX_NAME}' already contains {index.describe_index_stats().total_vector_count} vectors. Skipping upsert.")

    print("💾 Index stats after operations:", index.describe_index_stats())
    return index

def get_pinecone_index():
    """
    Connects to the Pinecone index.
    """
    print(f"DEBUG (get_pinecone_index): Connecting to index '{INDEX_NAME}'.")
    return pc.Index(INDEX_NAME)

def query_pinecone_index(api_key: str, user_query: str, n_results: int = 10) -> List[str]:
    """
    Queries the Pinecone index for relevant context.
    """
    print(f"DEBUG (query_pinecone_index): Querying index with '{user_query}'.")
    index = get_pinecone_index()
    
    # This will use the globally initialized embed_fn 
    query_embedding = embed_fn([user_query])[0]

    results = index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True
    )

    context_chunks = []
    if results.matches:
        for match in results.matches:
            chunk_content = match.metadata.get("text_content")
            if chunk_content:
                context_chunks.append(chunk_content)
    
    unique_chunks = []
    seen = set() # Use a set to keep track of seen chunks for uniqueness
    for chunk in context_chunks:
        # Ensure chunk is a non-empty string before adding
        if isinstance(chunk, str) and chunk.strip() and chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    
    # IMPORTANT: Return the list of unique chunks
    return unique_chunks