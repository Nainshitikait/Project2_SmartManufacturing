import pandas as pd
import joblib
import requests
from sentence_transformers import SentenceTransformer
import chromadb


# -----------------------------
# Load Trained ML Model
# -----------------------------

model = joblib.load(
    "models/random_forest_defect_model.pkl"
)


# -----------------------------
# New Sensor Input
# -----------------------------

sensor_data = pd.DataFrame([
    {
        "Surface_Roughness_Index": 3.0,
        "Vibration_Level": 6.0,
        "Temperature_Deviation": 1.5,
        "Dimensional_Deviation": 0.7,
        "Material_Stress_Indicator": 60.0
    }
])


# -----------------------------
# ML Prediction
# -----------------------------

prediction = model.predict(sensor_data)
predicted_defect = prediction[0]

probabilities = model.predict_proba(sensor_data)
confidence = probabilities.max()


# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="manufacturing_knowledge"
)


# -----------------------------
# RAG Query
# -----------------------------

query = f"""
What should I do for {predicted_defect}
when vibration is high?
"""


# -----------------------------
# Retrieve Knowledge
# -----------------------------

query_embedding = embedding_model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


retrieved_knowledge = "\n\n".join(
    results["documents"][0]
)


# -----------------------------
# LLM Recommendation
# -----------------------------

prompt = f"""
You are a manufacturing maintenance assistant.

Sensor-based ML prediction:
Defect Category: {predicted_defect}
Confidence: {confidence * 100:.2f}%

Relevant manufacturing knowledge:
{retrieved_knowledge}

Based only on the information above, provide:
1. A short explanation of the possible issue.
2. Recommended actions for the engineer.
3. A short safety/risk note.

Keep the answer practical and concise.
"""


response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False
    }
)


# -----------------------------
# Display Complete Pipeline
# -----------------------------

print("===================================")
print("SMART MANUFACTURING AI PIPELINE")
print("===================================")

print("\nSensor Input:")
print(sensor_data.to_string(index=False))

print("\nML Prediction:")
print("Defect Category:", predicted_defect)
print("Confidence:", f"{confidence * 100:.2f}%")

print("\nRAG Retrieved Knowledge:")
print(retrieved_knowledge)

print("\nLLM Recommendation:")

if response.status_code == 200:
    llm_result = response.json()
    print(llm_result["response"])
else:
    print("LLM Error:", response.text)

print("\n===================================")
print("PIPELINE COMPLETED")
print("===================================")