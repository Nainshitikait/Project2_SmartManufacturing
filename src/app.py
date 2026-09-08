import streamlit as st
import pandas as pd
import joblib
import requests
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Smart Manufacturing AI",
    page_icon="🏭",
    layout="wide"
)


# ---------------------------------
# Load ML Model
# ---------------------------------

model = joblib.load(
    "models/random_forest_defect_model.pkl"
)


# ---------------------------------
# Load Embedding Model
# ---------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------
# Connect to ChromaDB
# ---------------------------------

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="manufacturing_knowledge"
)


# ---------------------------------
# Application Title
# ---------------------------------

st.title("🏭 Smart Manufacturing AI Assistant")

st.write(
    "Enter manufacturing sensor values to predict "
    "defect severity and receive AI-based recommendations."
)


# ---------------------------------
# Sensor Inputs
# ---------------------------------

st.subheader("Sensor Input")

col1, col2 = st.columns(2)

with col1:
    surface_roughness = st.number_input(
        "Surface Roughness Index",
        value=3.0
    )

    vibration = st.number_input(
        "Vibration Level",
        value=6.0
    )

    temperature = st.number_input(
        "Temperature Deviation",
        value=1.5
    )

with col2:
    dimensional = st.number_input(
        "Dimensional Deviation",
        value=0.7
    )

    material_stress = st.number_input(
        "Material Stress Indicator",
        value=60.0
    )


# ---------------------------------
# Prediction Button
# ---------------------------------

if st.button("Predict Defect", type="primary"):

    sensor_data = pd.DataFrame([
        {
            "Surface_Roughness_Index": surface_roughness,
            "Vibration_Level": vibration,
            "Temperature_Deviation": temperature,
            "Dimensional_Deviation": dimensional,
            "Material_Stress_Indicator": material_stress
        }
    ])


    # ---------------------------------
    # ML Prediction
    # ---------------------------------

    prediction = model.predict(sensor_data)

    predicted_defect = prediction[0]

    probabilities = model.predict_proba(sensor_data)

    confidence = probabilities.max()


    st.subheader("ML Prediction")

    st.success(
        f"Defect Category: {predicted_defect}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # ---------------------------------
    # RAG Query
    # ---------------------------------

    query = f"""
    What should I do for {predicted_defect}
    when vibration is {vibration}?
    """


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


    # ---------------------------------
    # Display RAG Results
    # ---------------------------------

    st.subheader("Relevant Manufacturing Knowledge")

    st.write(retrieved_knowledge)


    # ---------------------------------
    # LLM Recommendation
    # ---------------------------------

    prompt = f"""
    You are a manufacturing maintenance assistant.

    Sensor-based ML prediction:
    Defect Category: {predicted_defect}
    Confidence: {confidence * 100:.2f}%

    Sensor values:
    Surface Roughness: {surface_roughness}
    Vibration: {vibration}
    Temperature Deviation: {temperature}
    Dimensional Deviation: {dimensional}
    Material Stress: {material_stress}

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


    # ---------------------------------
    # Display LLM Recommendation
    # ---------------------------------

    st.subheader("AI Recommendation")

    if response.status_code == 200:

        llm_result = response.json()

        st.write(
            llm_result["response"]
        )

    else:

        st.error(
            "Unable to connect to the LLM."
        )