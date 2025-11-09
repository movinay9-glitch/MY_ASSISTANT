import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# -------------------------
# Load API key
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# -------------------------
# Connect to Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("notes-assistant")

# -------------------------
# Load local embedding model
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Query
# -------------------------
query = input("Ask a question: ")
query_vector = model.encode(query).tolist()

results = index.query(vector=query_vector, top_k=3, include_metadata=True)

print("\n🔍 Top relevant answers:\n")
for match in results['matches']:
    print(f"- {match['metadata']['text']}")
    print(f"  📘 Source: {match['metadata']['source']} | Page: {match['metadata']['page']}\n")
