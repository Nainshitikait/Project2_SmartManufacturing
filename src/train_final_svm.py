import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    "data/manufacturing_sensor_data.csv"
)


# -----------------------------
# Features and Target
# -----------------------------

feature_columns = [
    "Surface_Roughness_Index",
    "Vibration_Level",
    "Temperature_Deviation",
    "Dimensional_Deviation",
    "Material_Stress_Indicator"
]

target_column = "Defect_Category"

X = df[feature_columns]
y = df[target_column]


# -----------------------------
# Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# Final SVM Model
# -----------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    )
])


# -----------------------------
# Train Model
# -----------------------------

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Save Model
# -----------------------------

joblib.dump(
    model,
    "models/svm_defect_model.pkl"
)


print("Final SVM model trained successfully.")
print("Model saved to:")
print("models/svm_defect_model.pkl")