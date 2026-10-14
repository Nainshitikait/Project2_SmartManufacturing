# Smart Manufacturing: RAG-Based Intelligent Defect Detection



## Project Overview



This project develops an intelligent defect detection and troubleshooting assistance system for smart manufacturing. It combines machine learning classification with Retrieval-Augmented Generation (RAG) to help analyze manufacturing sensor data, identify potential defect severity, and retrieve relevant troubleshooting guidance.



The application provides an interactive Streamlit dashboard for analyzing sensor records, comparing machine learning models, and exploring recommendations based on a manufacturing knowledge base.



**Note:** The current dataset is synthetic. Model performance on synthetic data does not establish performance on real factory data.



## Key Features



- **Defect Classification:** Classify manufacturing records into No Defect, Minor, Moderate, and Critical categories.

- **Model Comparison:** Evaluate Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine (SVM).

- **Model Evaluation:** Generate classification reports, confusion matrices, and comparison metrics.

- **Final SVM Model:** Save and reuse the trained SVM model for prediction.

- **RAG-Based Assistance:** Retrieve relevant manufacturing troubleshooting information from a knowledge base.

- **Semantic Search:** Use Sentence Transformers embeddings and ChromaDB for document retrieval.

- **Local LLM Integration:** Use Ollama for generating responses from retrieved context.

- **Interactive Dashboard:** Explore predictions, sensor data, and troubleshooting recommendations through Streamlit.

- **Batch Processing:** Process multiple manufacturing records and summarize observed defect severity.



## Technology Stack



- Python

- Pandas and NumPy

- Scikit-learn

- Sentence Transformers

- ChromaDB

- Ollama

- Streamlit

- Matplotlib

- Joblib



## Machine Learning Models



The project compares four classification algorithms using a stratified train-test split.



| Model | Accuracy | Weighted F1 Score |

|---|---:|---:|

| Logistic Regression | 83.25% | 83.14% |

| Random Forest | 90.75% | 90.77% |

| Gradient Boosting | 91.45% | 91.46% |

| SVM | 93.10% | 93.11% |



In the current evaluation, SVM achieves the highest accuracy and weighted F1 score among the four tested models.



These metrics were measured on the project's synthetic dataset and should not be interpreted as validated real-world manufacturing performance.



## Sensor Features



The synthetic dataset uses five sensor-related features:



- Surface Roughness

- Vibration Level

- Temperature Deviation

- Dimensional Deviation

- Material Stress Indicator



The target classes are No Defect, Minor, Moderate, and Critical.



## RAG Pipeline



1. Load manufacturing troubleshooting documents.

2. Split the documents into smaller text chunks.

3. Generate semantic embeddings using `all-MiniLM-L6-v2`.

4. Store and retrieve relevant chunks using ChromaDB.

5. Apply relevance and severity-aware ranking.

6. Use retrieved context to support troubleshooting responses generated through Ollama.



The troubleshooting recommendations depend on the contents of the knowledge base and the quality of retrieval and generation.



## Running the Application



### Prerequisites



- Python and pip

- Git

- Ollama, with the required language model downloaded

- The project's Python dependencies



### 1. Clone the repository



```bash

git clone https://github.com/Nainshitikait/Project2\_SmartManufacturing.git

cd Project2\_SmartManufacturing

```



### 2. Create a virtual environment



On Windows:



```bash

python -m venv .venv

.venv\Scripts\activate

```



### 3. Install dependencies



If the repository contains `requirements.txt`, install dependencies with:



```bash

python -m pip install -r requirements.txt

```



### 4. Prepare Ollama



Start Ollama and download the language model configured by the application. For example, if the application uses `llama3.1:8b`:



```bash

ollama pull llama3.1:8b

```



Make sure the model name matches the configuration in your code.



### 5. Launch the dashboard



From the project root, run:



```bash

python -m streamlit run src/app.py

```



Open the local URL printed in the terminal, usually `http://localhost:8501`.



The first run may download the Sentence Transformers embedding model if it is not already available locally.



## Evaluation Artifacts



The project includes model comparison results, classification reports, and confusion matrix visualizations to support model evaluation.



The saved SVM model can be found at:



`models/svm\_defect\_model.pkl`



## Deployment Status



The current application is designed for local execution. Azure cloud deployment is a potential future enhancement and is not represented as a completed deployment.



## Future Improvements



- Evaluate the models using real, labeled manufacturing data.

- Validate predictions across different machines and operating conditions.

- Expand the troubleshooting knowledge base.

- Improve automated testing and evaluation.

- Explore Azure deployment and cloud-based monitoring.



## Responsible Use



This system is intended as a prototype for decision support and experimentation. Predictions and troubleshooting recommendations should be validated by qualified personnel before being used in manufacturing operations.



## Project Information



**Project:** RAG-Based Intelligent Defect Detection and Knowledge Assistance for Smart Manufacturing



**Focus Areas:** Machine Learning, Retrieval-Augmented Generation, Natural Language Processing, and Interactive Data Visualization.
