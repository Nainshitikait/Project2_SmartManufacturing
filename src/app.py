import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import chromadb
import time

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Manufacturing Defect Detection",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/manufacturing_sensor_data.csv"
MODEL_PATH = "models/svm_defect_model.pkl"
CHROMA_PATH = "data/chroma_db"


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# LOAD SENSOR DATA
# ============================================================

@st.cache_data
def load_sensor_data():
    return pd.read_csv(DATA_PATH)


sensor_df = load_sensor_data()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ============================================================
# CONNECT TO CHROMA KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name="manufacturing_knowledge"
    )

    return collection


collection = load_knowledge_base()


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "sensor_index" not in st.session_state:
    st.session_state.sensor_index = 0

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "batch_predictions" not in st.session_state:
    st.session_state.batch_predictions = []

if "current_batch_start" not in st.session_state:
    st.session_state.current_batch_start = 1

if "last_completed_batch" not in st.session_state:
    st.session_state.last_completed_batch = None


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🏭 Smart Manufacturing Defect Detection")

st.write(
    "Automatic sensor monitoring with ML-based defect detection, "
    "RAG-based knowledge retrieval and LLM recommendations."
)


# ============================================================
# MANUFACTURING CONTEXT
# ============================================================

st.subheader("🏭 Manufacturing Context")

context_col1, context_col2, context_col3 = st.columns(3)

with context_col1:
    st.info(
        "**Industry**\n\n"
        "Automotive Manufacturing"
    )

with context_col2:
    st.info(
        "**Monitored Component**\n\n"
        "Automotive Component"
    )

with context_col3:
    st.info(
        "**Application**\n\n"
        "Defect Detection & Quality Monitoring"
    )


# ============================================================
# SENSOR MONITORING FIELDS
# ============================================================

st.subheader("📊 Sensor Monitoring Fields")

field_col1, field_col2, field_col3, field_col4, field_col5 = st.columns(5)

with field_col1:
    st.write("**Surface Quality**")
    st.caption("Surface Roughness")
    st.caption("Monitors surface finish condition")

with field_col2:
    st.write("**Machine Health**")
    st.caption("Vibration Level")
    st.caption("Monitors machine vibration")

with field_col3:
    st.write("**Thermal Condition**")
    st.caption("Temperature Deviation")
    st.caption("Monitors temperature variation")

with field_col4:
    st.write("**Dimensional Quality**")
    st.caption("Dimensional Deviation")
    st.caption("Monitors dimensional accuracy")

with field_col5:
    st.write("**Material Integrity**")
    st.caption("Material Stress")
    st.caption("Monitors material stress condition")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("System Status")

st.sidebar.success("ML Model Loaded")
st.sidebar.success("Sensor Data Loaded")
st.sidebar.success("Knowledge Base Connected")
st.sidebar.info("LLM: Llama 3.1 8B")

st.sidebar.divider()

st.sidebar.subheader("📡 Monitoring Control")

if st.session_state.monitoring:

    st.sidebar.success("🟢 Monitoring Active")

    if st.sidebar.button(
        "⏹ Stop Monitoring",
        type="secondary"
    ):
        st.session_state.monitoring = False
        st.rerun()

else:

    st.sidebar.warning("🔴 Monitoring Stopped")

    if st.sidebar.button(
        "▶ Start Monitoring",
        type="primary"
    ):
        st.session_state.monitoring = True
        st.rerun()


# ============================================================
# AUTOMATIC SENSOR MONITORING
# ============================================================

st.subheader("📡 Automatic Sensor Monitoring")

if st.session_state.monitoring:

    st.success(
        "🟢 Live monitoring is active. "
        "Sensor readings are being processed automatically."
    )

    st.info(
        "The system automatically processes each sensor reading "
        "and moves to the next reading every 5 seconds."
    )

else:

    st.warning(
        "🔴 Monitoring is stopped. "
        "Click 'Start Monitoring' from the sidebar to begin."
    )


# ============================================================
# CURRENT SENSOR READING
# ============================================================

current_index = st.session_state.sensor_index

sensor_row = sensor_df.iloc[current_index]

reading_number = current_index + 1

total_readings = len(sensor_df)


# ============================================================
# READING BATCH
# ============================================================

batch_start = ((reading_number - 1) // 10) * 10 + 1

batch_end = min(
    batch_start + 9,
    total_readings
)

reading_position_in_batch = (
    reading_number - batch_start + 1
)

batch_size = (
    batch_end - batch_start + 1
)


# Save completed batch when moving to next batch.

if st.session_state.current_batch_start != batch_start:

    if st.session_state.batch_predictions:

        st.session_state.last_completed_batch = {
            "batch_start":
                st.session_state.current_batch_start,

            "batch_end":
                st.session_state.current_batch_start
                + len(st.session_state.batch_predictions)
                - 1,

            "predictions":
                st.session_state.batch_predictions.copy()
        }

    st.session_state.batch_predictions = []

    st.session_state.current_batch_start = batch_start


st.divider()

batch_col1, batch_col2, batch_col3 = st.columns(3)

with batch_col1:
    st.metric(
        "📦 Current Batch",
        f"{batch_start}–{batch_end}"
    )

with batch_col2:
    st.metric(
        "📍 Current Reading",
        f"{reading_number}"
    )

with batch_col3:
    st.metric(
        "🔢 Position in Batch",
        f"{reading_position_in_batch}/{batch_size}"
    )

st.progress(
    reading_position_in_batch / batch_size
)

st.caption(
    f"Processing Reading #{reading_number} of "
    f"{total_readings} | Batch {batch_start}–{batch_end}"
)


# ============================================================
# DISPLAY SENSOR VALUES
# ============================================================

st.subheader("📈 Current Sensor Values")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Surface Roughness",
        f"{sensor_row['Surface_Roughness_Index']:.3f}"
    )

with col2:
    st.metric(
        "Vibration",
        f"{sensor_row['Vibration_Level']:.3f}"
    )

with col3:
    st.metric(
        "Temperature Deviation",
        f"{sensor_row['Temperature_Deviation']:.3f}"
    )

with col4:
    st.metric(
        "Dimensional Deviation",
        f"{sensor_row['Dimensional_Deviation']:.3f}"
    )

with col5:
    st.metric(
        "Material Stress",
        f"{sensor_row['Material_Stress_Indicator']:.3f}"
    )


# ============================================================
# SENSOR DATAFRAME FOR ML MODEL
# ============================================================

sensor_data = pd.DataFrame([
    {
        "Surface_Roughness_Index":
            sensor_row["Surface_Roughness_Index"],

        "Vibration_Level":
            sensor_row["Vibration_Level"],

        "Temperature_Deviation":
            sensor_row["Temperature_Deviation"],

        "Dimensional_Deviation":
            sensor_row["Dimensional_Deviation"],

        "Material_Stress_Indicator":
            sensor_row["Material_Stress_Indicator"]
    }
])


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_sensor_reading(
    sensor_row,
    sensor_data
):

    prediction = model.predict(
        sensor_data
    )[0]

    probabilities = model.predict_proba(
        sensor_data
    )[0]

    confidence = float(
        np.max(probabilities)
    )


    # --------------------------------------------------------
    # SENSOR VALUES
    # --------------------------------------------------------

    vibration_value = (
        sensor_row["Vibration_Level"]
    )

    temperature_value = (
        sensor_row["Temperature_Deviation"]
    )

    roughness_value = (
        sensor_row["Surface_Roughness_Index"]
    )

    dimensional_value = (
        sensor_row["Dimensional_Deviation"]
    )

    stress_value = (
        sensor_row["Material_Stress_Indicator"]
    )


    # --------------------------------------------------------
    # RAG QUERY
    # --------------------------------------------------------

    rag_query = f"""

    Automotive manufacturing defect detection.

    Industry:
    Automotive Manufacturing

    Component:
    Automotive Component

    Application:
    Defect Detection and Quality Monitoring

    Predicted defect severity:
    {prediction}

    Sensor conditions:

    Surface Quality / Surface Roughness:
    {roughness_value:.2f}

    Machine Health / Vibration Level:
    {vibration_value:.2f}

    Thermal Condition / Temperature Deviation:
    {temperature_value:.2f}

    Dimensional Quality / Dimensional Deviation:
    {dimensional_value:.2f}

    Material Integrity / Material Stress:
    {stress_value:.2f}

    Find manufacturing knowledge specifically relevant
    to the predicted defect severity.

    What could be the possible automotive manufacturing issue
    and what corrective action should be taken?

    """


    # --------------------------------------------------------
    # CREATE EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        rag_query
    ).tolist()


    # --------------------------------------------------------
    # RETRIEVE KNOWLEDGE
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    candidate_documents = results.get(
        "documents",
        [[]]
    )[0]


    # --------------------------------------------------------
    # SEVERITY-AWARE RERANKING
    # --------------------------------------------------------

    severity_keywords = {

        "Critical Defect": [
            "critical",
            "severe",
            "failure",
            "shutdown",
            "immediate",
            "urgent"
        ],

        "Moderate Defect": [
            "moderate",
            "abnormal",
            "maintenance",
            "inspection",
            "corrective"
        ],

        "Minor Defect": [
            "minor",
            "slight",
            "monitor",
            "adjustment",
            "inspection"
        ],

        "No Defect": [
            "no defect",
            "normal",
            "normal operation",
            "routine monitoring",
            "preventive maintenance"
        ]
    }

    keywords = severity_keywords.get(
        prediction,
        []
    )

    def severity_match_score(document):

        text = document.lower()

        score = 0

        for keyword in keywords:

            if keyword.lower() in text:
                score += 1

        return score

    ranked_documents = sorted(
        candidate_documents,
        key=severity_match_score,
        reverse=True
    )

    retrieved_documents = ranked_documents[:3]


    # --------------------------------------------------------
    # LLM RECOMMENDATION
    # --------------------------------------------------------

    knowledge_context = "\n\n".join(
        retrieved_documents
    )

    prompt = f"""

You are an intelligent manufacturing assistant.

Industry:

Automotive Manufacturing


Component:

Automotive Component


Application:

Defect Detection and Quality Monitoring


A machine monitoring system detected:


Defect classification:

{prediction}


Prediction confidence:

{confidence * 100:.2f}%


Sensor readings:


Surface Quality / Surface Roughness:

{roughness_value:.3f}


Machine Health / Vibration:

{vibration_value:.3f}


Thermal Condition / Temperature Deviation:

{temperature_value:.3f}


Dimensional Quality / Dimensional Deviation:

{dimensional_value:.3f}


Material Integrity / Material Stress:

{stress_value:.3f}


Relevant manufacturing knowledge:


{knowledge_context}


Provide a concise manufacturing recommendation.


Structure your answer as:


Possible Issue:

Explain the likely issue.


Recommended Action:

Give practical corrective actions.


Risk:

Explain the possible production or machine risk.


Keep the answer concise and base it on the sensor readings
and retrieved manufacturing knowledge.

Do not contradict the predicted defect classification.
"""


    # --------------------------------------------------------
    # CALL LLM
    # --------------------------------------------------------

    llm_result = ""

    try:

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False
            },

            timeout=120
        )

        if response.status_code == 200:

            llm_result = response.json().get(
                "response",
                ""
            )

        else:

            llm_result = (
                "Unable to get a response from the LLM."
            )

    except requests.exceptions.RequestException:

        llm_result = (
            "Unable to connect to the LLM. "
            "Make sure Ollama is running."
        )

    return (
        prediction,
        confidence,
        retrieved_documents,
        llm_result
    )
    # ============================================================
# MANUAL ANALYSIS BUTTON
# ============================================================

if not st.session_state.monitoring:

    if st.button(
        "🔍 Analyze Current Sensor Reading",
        type="primary"
    ):

        with st.spinner(
            "Analyzing sensor reading..."
        ):

            (
                prediction,
                confidence,
                retrieved_documents,
                llm_result
            ) = analyze_sensor_reading(
                sensor_row,
                sensor_data
            )

        st.session_state.last_prediction = (
            prediction,
            confidence,
            retrieved_documents,
            llm_result
        )


# ============================================================
# AUTOMATIC MONITORING ANALYSIS
# ============================================================

if st.session_state.monitoring:

    with st.spinner(
        f"Processing Reading #{reading_number}..."
    ):

        (
            prediction,
            confidence,
            retrieved_documents,
            llm_result
        ) = analyze_sensor_reading(
            sensor_row,
            sensor_data
        )

    st.session_state.last_prediction = (
        prediction,
        confidence,
        retrieved_documents,
        llm_result
    )

    # Store prediction for current batch
    st.session_state.batch_predictions.append(
        prediction
    )


# ============================================================
# DISPLAY ANALYSIS RESULT
# ============================================================

if st.session_state.last_prediction is not None:

    (
        prediction,
        confidence,
        retrieved_documents,
        llm_result
    ) = st.session_state.last_prediction


    # ========================================================
    # ML RESULT
    # ========================================================

    st.divider()

    st.subheader("🤖 ML Defect Detection")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == "No Defect":

            st.success(
                f"Prediction: {prediction}"
            )

        elif prediction == "Minor Defect":

            st.warning(
                f"Prediction: {prediction}"
            )

        elif prediction == "Moderate Defect":

            st.warning(
                f"Prediction: {prediction}"
            )

        else:

            st.error(
                f"Prediction: {prediction}"
            )


    with result_col2:

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    st.divider()

    st.subheader(
        "📚 Manufacturing Knowledge Retrieval"
    )

    if retrieved_documents:

        for i, document in enumerate(
            retrieved_documents,
            start=1
        ):

            with st.expander(
                f"Retrieved Knowledge {i}"
            ):

                st.write(document)

    else:

        st.warning(
            "No relevant manufacturing knowledge "
            "was retrieved."
        )


    # ========================================================
    # LLM RECOMMENDATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 AI Manufacturing Recommendation"
    )

    st.info(llm_result)


# ============================================================
# BATCH SUMMARY
# ============================================================

if (
    st.session_state.batch_predictions
    and st.session_state.monitoring
):

    st.divider()

    st.subheader(
        "📦 Current Batch Summary"
    )

    predictions = (
        st.session_state.batch_predictions
    )

    no_defect_count = predictions.count(
        "No Defect"
    )

    minor_count = predictions.count(
        "Minor Defect"
    )

    moderate_count = predictions.count(
        "Moderate Defect"
    )

    critical_count = predictions.count(
        "Critical Defect"
    )


    # --------------------------------------------------------
    # BATCH COUNTS
    # --------------------------------------------------------

    summary_col1, summary_col2, summary_col3, summary_col4 = (
        st.columns(4)
    )

    with summary_col1:

        st.metric(
            "No Defect",
            no_defect_count
        )

    with summary_col2:

        st.metric(
            "Minor Defect",
            minor_count
        )

    with summary_col3:

        st.metric(
            "Moderate Defect",
            moderate_count
        )

    with summary_col4:

        st.metric(
            "Critical Defect",
            critical_count
        )


    # --------------------------------------------------------
    # OVERALL BATCH SEVERITY
    # --------------------------------------------------------

    severity_priority = {

        "No Defect": 0,

        "Minor Defect": 1,

        "Moderate Defect": 2,

        "Critical Defect": 3
    }


    overall_batch_status = max(
        predictions,
        key=lambda x: severity_priority.get(
            x,
            0
        )
    )


    st.write("### Overall Batch Status")


    if overall_batch_status == "No Defect":

        st.success(
            f"🟢 {overall_batch_status}"
        )

    elif overall_batch_status == "Minor Defect":

        st.warning(
            f"🟡 {overall_batch_status}"
        )

    elif overall_batch_status == "Moderate Defect":

        st.warning(
            f"🟠 {overall_batch_status}"
        )

    else:

        st.error(
            f"🔴 {overall_batch_status}"
        )


    st.caption(
        f"Processed {len(predictions)} "
        f"of {batch_size} readings in "
        f"Batch {batch_start}–{batch_end}."
    )


# ============================================================
# PREVIOUS COMPLETED BATCH SUMMARY
# ============================================================

if (
    st.session_state.last_completed_batch
    is not None
):

    completed_batch = (
        st.session_state.last_completed_batch
    )

    completed_predictions = (
        completed_batch["predictions"]
    )

    completed_start = (
        completed_batch["batch_start"]
    )

    completed_end = (
        completed_batch["batch_end"]
    )


    st.divider()

    st.subheader(
        "📊 Previous Completed Batch"
    )

    previous_col1, previous_col2, previous_col3, previous_col4 = (
        st.columns(4)
    )

    with previous_col1:

        st.metric(
            "No Defect",
            completed_predictions.count(
                "No Defect"
            )
        )

    with previous_col2:

        st.metric(
            "Minor Defect",
            completed_predictions.count(
                "Minor Defect"
            )
        )

    with previous_col3:

        st.metric(
            "Moderate Defect",
            completed_predictions.count(
                "Moderate Defect"
            )
        )

    with previous_col4:

        st.metric(
            "Critical Defect",
            completed_predictions.count(
                "Critical Defect"
            )
        )


    completed_status = max(
        completed_predictions,
        key=lambda x: severity_priority.get(
            x,
            0
        )
    )


    st.info(
        f"Batch {completed_start}–{completed_end} "
        f"Overall Status: **{completed_status}**"
    )


# ============================================================
# MONITORING INFORMATION
# ============================================================

if st.session_state.monitoring:

    st.divider()

    st.subheader("🔄 Live Monitoring")

    st.write(
        "The system processes each sensor reading individually "
        "while organizing readings into batches of 10."
    )

    st.caption(
        f"Current batch: {batch_start}–{batch_end}"
    )

    st.caption(
        f"Batch progress: "
        f"{len(st.session_state.batch_predictions)}/"
        f"{batch_size} readings processed"
    )

    st.caption(
        "Monitoring interval: 5 seconds"
    )


    # --------------------------------------------------------
    # MOVE TO NEXT SENSOR READING
    # --------------------------------------------------------

    time.sleep(5)


    if (
        st.session_state.sensor_index
        < len(sensor_df) - 1
    ):

        st.session_state.sensor_index += 1

    else:

        st.session_state.sensor_index = 0

        # Reset batch tracking when dataset restarts
        st.session_state.batch_predictions = []

        st.session_state.current_batch_start = 1

        st.session_state.last_completed_batch = None


    st.rerun()


# ============================================================
# MANUAL NEXT SENSOR READING
# ============================================================

if not st.session_state.monitoring:

    st.divider()

    if st.button(
        "➡️ Load Next Sensor Reading"
    ):

        if (
            st.session_state.sensor_index
            < len(sensor_df) - 1
        ):

            st.session_state.sensor_index += 1

        else:

            st.session_state.sensor_index = 0

        st.session_state.last_prediction = None

        st.rerun()


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander(
    "📊 Sensor Dataset Information"
):

    st.write(
        f"Total available sensor readings: "
        f"{len(sensor_df)}"
    )

    st.write(
        "The application automatically reads "
        "sensor records from the manufacturing dataset."
    )

    st.write(
        "Each sensor reading is processed individually "
        "through the ML defect detection, RAG knowledge "
        "retrieval and LLM recommendation pipeline."
    )

    st.write(
        "For monitoring organization, readings are grouped "
        "into batches of 10: 1–10, 11–20, 21–30, and so on."
    )