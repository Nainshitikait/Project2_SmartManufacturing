import numpy as np
import pandas as pd


# -----------------------------
# Reproducible Results
# -----------------------------

np.random.seed(42)


# -----------------------------
# Number of Records
# -----------------------------

n = 10000


# -----------------------------
# Generate Sensor Features
# -----------------------------

surface_roughness = np.random.normal(2.5, 0.8, n)

vibration_level = np.random.normal(5.0, 1.5, n)

temperature_deviation = np.random.normal(0.0, 2.0, n)

dimensional_deviation = np.random.normal(0.0, 0.5, n)

material_stress = np.random.normal(50.0, 10.0, n)


# -----------------------------
# Create Defect Score
# -----------------------------

defect_score = (
    surface_roughness * 0.25
    + vibration_level * 0.20
    + abs(temperature_deviation) * 0.15
    + abs(dimensional_deviation) * 0.20
    + material_stress * 0.20
)


# -----------------------------
# Convert Score to Defect Class
# -----------------------------

defect_category = np.where(
    defect_score < 13.5,
    "No Defect",
    np.where(
        defect_score < 14.5,
        "Minor Defect",
        np.where(
            defect_score < 15.5,
            "Moderate Defect",
            "Critical Defect"
        )
    )
)


# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "Surface_Roughness_Index": surface_roughness,
    "Vibration_Level": vibration_level,
    "Temperature_Deviation": temperature_deviation,
    "Dimensional_Deviation": dimensional_deviation,
    "Material_Stress_Indicator": material_stress,
    "Defect_Category": defect_category
})


# -----------------------------
# Save Dataset
# -----------------------------

output_path = "data/manufacturing_sensor_data.csv"

df.to_csv(output_path, index=False)


# -----------------------------
# Display Dataset Information
# -----------------------------

print("Synthetic manufacturing dataset created successfully!")

print("Rows:", len(df))

print("Columns:", len(df.columns))


print("\nDataset Preview:")

print(df.head())


print("\nDefect Category Distribution:")

print(df["Defect_Category"].value_counts())