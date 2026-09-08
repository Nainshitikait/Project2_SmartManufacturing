import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# ChromaDB
# -----------------------------

chroma_path = "data/chroma_db"

client = chromadb.PersistentClient(
    path=chroma_path
)

collection = client.get_or_create_collection(
    name="manufacturing_knowledge"
)


# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# User Query
# -----------------------------

query = "What should I do if vibration is high?"


# -----------------------------
# Convert Query to Embedding
# -----------------------------

query_embedding = embedding_model.encode(
    [query]
).tolist()


# -----------------------------
# Search Knowledge Base
# -----------------------------

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


# -----------------------------
# Display Retrieved Knowledge
# -----------------------------

print("\nUser Query:")
print(query)

print("\nRelevant Manufacturing Knowledge:")

for i, document in enumerate(
    results["documents"][0],
    start=1
):
    print(f"\n--- Result {i} ---")
    print(document)