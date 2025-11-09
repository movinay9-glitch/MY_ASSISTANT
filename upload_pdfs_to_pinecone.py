import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import PyPDF2
from tqdm import tqdm  # for nice progress bar

# -------------------------
# 1️⃣ Load environment variables
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PDF_FOLDER = "PDF_NOTES"
INDEX_NAME = "notes-assistant"

# -------------------------
# 2️⃣ Initialize Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)  # ✅ MISSING PART ADDED

# Delete old index if exists (optional reset)
if INDEX_NAME in pc.list_indexes().names():
    print("🗑️ Deleting old index to fix dimension mismatch...")
    pc.delete_index(INDEX_NAME)

# Create new index with correct dimension
pc.create_index(
    name=INDEX_NAME,
    dimension=384,  # ✅ correct for all-MiniLM-L6-v2
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
index = pc.Index(INDEX_NAME)

# -------------------------
# 3️⃣ Load embedding model
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# 4️⃣ Extract text from PDFs
# -------------------------
all_chunks = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Split into 500-character chunks
                    for i in range(0, len(text), 500):
                        chunk = text[i:i+500].strip()
                        if chunk:
                            all_chunks.append({
                                "text": chunk,
                                "source": filename,
                                "page": page_num + 1
                            })

print(f"ℹ️ Total chunks to upload: {len(all_chunks)}")

# -------------------------
# 5️⃣ Generate embeddings and upload
# -------------------------
batch_size = 20
for i in tqdm(range(0, len(all_chunks), batch_size), desc="Uploading to Pinecone"):
    batch = all_chunks[i:i+batch_size]
    texts = [chunk["text"] for chunk in batch]
    embeddings = model.encode(texts).tolist()

    to_upsert = []
    for j, chunk in enumerate(batch):
        to_upsert.append((
            f"id-{i+j}",
            embeddings[j],
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"]
            }
        ))
    index.upsert(vectors=to_upsert)

print("✅ All PDF chunks uploaded to Pinecone successfully!")
