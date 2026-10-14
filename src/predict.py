import pandas as pd
import joblib


# -----------------------------
# Load Final SVM Model
# -----------------------------

model = joblib.load(
    "models/svm_defect_model.pkl"
)


# -----------------------------
# New Sensor Reading
# -----------------------------

new_sensor_data = pd.DataFrame([
    {
        "Surface_Roughness_Index": 3.0,
        "Vibration_Level": 6.0,
        "Temperature_Deviation": 1.5,
        "Dimensional_Deviation": 0.7,
        "Material_Stress_Indicator": 60.0
    }
])


# -----------------------------
# Predict Defect Category
# -----------------------------

prediction = model.predict(
    new_sensor_data
)

predicted_class = prediction[0]


# -----------------------------
# Prediction Probabilities
# -----------------------------

probabilities = model.predict_proba(
    new_sensor_data
)

class_names = model.classes_

confidence = probabilities.max()


# -----------------------------
# Display Prediction
# -----------------------------

print("New Sensor Reading:")
print(new_sensor_data)

print("\nPredicted Defect Category:")
print(predicted_class)

print("\nPrediction Confidence:")
print(f"{confidence * 100:.2f}%")

print("\nClass Probabilities:")

for class_name, probability in zip(
    class_names,
    probabilities[0]
):
    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )