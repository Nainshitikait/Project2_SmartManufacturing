import os
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# File Paths
# -----------------------------

document_path = "documents/manufacturing_troubleshooting.txt"

chroma_path = "data/chroma_db"


# -----------------------------
# Read Manufacturing Document
# -----------------------------

with open(
    document_path,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# -----------------------------
# Split Document into Chunks
# -----------------------------

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print("Document loaded successfully.")

print("Number of chunks:", len(chunks))


# -----------------------------
# Load Embedding Model
# -----------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# Create ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path=chroma_path
)


collection = client.get_or_create_collection(
    name="manufacturing_knowledge"
)


# -----------------------------
# Generate Embeddings
# -----------------------------

print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    chunks
).tolist()


# -----------------------------
# Store in ChromaDB
# -----------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


# -----------------------------
# Confirmation
# -----------------------------

print("\nKnowledge base created successfully!")

print(
    "Documents stored:",
    collection.count()
)

print(
    "Database location:",
    os.path.abspath(chroma_path)
)