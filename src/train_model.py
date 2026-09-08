import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/manufacturing_sensor_data.csv")


# -----------------------------
# Features and Target
# -----------------------------

X = df[
    [
        "Surface_Roughness_Index",
        "Vibration_Level",
        "Temperature_Deviation",
        "Dimensional_Deviation",
        "Material_Stress_Indicator"
    ]
]

y = df["Defect_Category"]


# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------
# Train Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Make Predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Evaluate Model
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Defect Classification Confusion Matrix")
plt.tight_layout()
plt.show()


# -----------------------------
# Save Trained Model
# -----------------------------

joblib.dump(
    model,
    "models/random_forest_defect_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Saved to: models/random_forest_defect_model.pkl"
)