import numpy as np
import pandas as pd

np.random.seed(42)

# Number of manufacturing records
n = 10000

# ---------------------------------------------------------
# 1. Generate latent manufacturing severity
# ---------------------------------------------------------
# This represents the underlying operating condition of
# the manufacturing process.
#
# 0 = normal
# 1 = increasingly severe operating condition
# ---------------------------------------------------------

severity = np.random.beta(2.2, 4.0, n)

# Add a small number of high-severity operating conditions
critical_indices = np.random.choice(
    n,
    size=int(n * 0.08),
    replace=False
)

severity[critical_indices] = np.random.uniform(
    0.75, 1.0, len(critical_indices)
)

# ---------------------------------------------------------
# 2. Generate correlated sensor measurements
# ---------------------------------------------------------

# Surface Roughness
surface_roughness = (
    1.8
    + severity * 2.8
    + np.random.normal(0, 0.35, n)
)

# Vibration
vibration_level = (
    3.0
    + severity * 5.0
    + np.random.normal(0, 0.65, n)
)

# Temperature deviation
temperature_deviation = (
    severity * 4.5
    + np.random.normal(0, 0.9, n)
)

# Randomly allow temperature to deviate in either direction
temperature_deviation *= np.random.choice(
    [-1, 1],
    size=n,
    p=[0.35, 0.65]
)

# Dimensional deviation
dimensional_deviation = (
    severity * 1.0
    + np.random.normal(0, 0.18, n)
)

dimensional_deviation *= np.random.choice(
    [-1, 1],
    size=n,
    p=[0.45, 0.55]
)

# Material stress
material_stress = (
    38
    + severity * 32
    + np.random.normal(0, 5, n)
)

# ---------------------------------------------------------
# 3. Add realistic interaction effects
# ---------------------------------------------------------

# High vibration can increase temperature
temperature_deviation += (
    np.maximum(vibration_level - 6.0, 0) * 0.15
)

# High stress can increase surface roughness
surface_roughness += (
    np.maximum(material_stress - 55.0, 0) * 0.012
)

# High vibration can increase dimensional deviation
dimensional_deviation += (
    np.maximum(vibration_level - 6.0, 0)
    * np.random.normal(0.025, 0.008, n)
)

# ---------------------------------------------------------
# 4. Keep sensor values within realistic ranges
# ---------------------------------------------------------

surface_roughness = np.clip(
    surface_roughness,
    0.5,
    6.0
)

vibration_level = np.clip(
    vibration_level,
    0.5,
    10.0
)

temperature_deviation = np.clip(
    temperature_deviation,
    -6.0,
    6.0
)

dimensional_deviation = np.clip(
    dimensional_deviation,
    -1.5,
    1.5
)

material_stress = np.clip(
    material_stress,
    25.0,
    80.0
)

# ---------------------------------------------------------
# 5. Convert sensor measurements into risk scores
# ---------------------------------------------------------

surface_risk = np.clip(
    (surface_roughness - 1.5) / 4.5,
    0,
    1
)

vibration_risk = np.clip(
    (vibration_level - 2.5) / 7.5,
    0,
    1
)

temperature_risk = np.clip(
    np.abs(temperature_deviation) / 6.0,
    0,
    1
)

dimensional_risk = np.clip(
    np.abs(dimensional_deviation) / 1.5,
    0,
    1
)

stress_risk = np.clip(
    (material_stress - 25.0) / 55.0,
    0,
    1
)

# ---------------------------------------------------------
# 6. Calculate overall manufacturing condition score
# ---------------------------------------------------------

condition_score = (
    surface_risk * 0.20
    + vibration_risk * 0.25
    + temperature_risk * 0.15
    + dimensional_risk * 0.20
    + stress_risk * 0.20
)

# Add small measurement uncertainty
condition_score += np.random.normal(
    0,
    0.015,
    n
)

condition_score = np.clip(
    condition_score,
    0,
    1
)

# ---------------------------------------------------------
# 7. Automatically determine defect severity
# ---------------------------------------------------------
#
# The thresholds are calculated from the generated
# manufacturing condition scores rather than manually
# assigning individual records.
#
# 50%  -> No Defect
# 25%  -> Minor Defect
# 15%  -> Moderate Defect
# 10%  -> Critical Defect
# ---------------------------------------------------------

no_defect_limit = np.quantile(
    condition_score,
    0.50
)

minor_defect_limit = np.quantile(
    condition_score,
    0.75
)

moderate_defect_limit = np.quantile(
    condition_score,
    0.90
)

defect_category = np.select(
    [
        condition_score <= no_defect_limit,
        condition_score <= minor_defect_limit,
        condition_score <= moderate_defect_limit
    ],
    [
        "No Defect",
        "Minor Defect",
        "Moderate Defect"
    ],
    default="Critical Defect"
)

# ---------------------------------------------------------
# 8. Create final dataset
# ---------------------------------------------------------

df = pd.DataFrame({
    "Surface_Roughness_Index": surface_roughness,
    "Vibration_Level": vibration_level,
    "Temperature_Deviation": temperature_deviation,
    "Dimensional_Deviation": dimensional_deviation,
    "Material_Stress_Indicator": material_stress,
    "Defect_Category": defect_category
})

# ---------------------------------------------------------
# 9. Save dataset
# ---------------------------------------------------------

output_path = "data/manufacturing_sensor_data.csv"

df.to_csv(
    output_path,
    index=False
)

# ---------------------------------------------------------
# 10. Display dataset information
# ---------------------------------------------------------

print("Dataset generated successfully!")
print(f"Saved to: {output_path}")

print("\nDataset Shape:")
print(df.shape)

print("\nDefect Category Distribution:")
print(df["Defect_Category"].value_counts())

print("\nSensor Statistics:")
print(df.describe())

print("\nAutomatically Calculated Severity Limits:")
print(f"No Defect limit: {no_defect_limit:.3f}")
print(f"Minor Defect limit: {minor_defect_limit:.3f}")
print(f"Moderate Defect limit: {moderate_defect_limit:.3f}")