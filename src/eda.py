import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/manufacturing_sensor_data.csv")

# Create folder for EDA graphs
os.makedirs("data/eda_plots", exist_ok=True)

# --------------------------------
# 1. Basic Dataset Information
# --------------------------------

print("SMART MANUFACTURING - EDA")
print("=" * 40)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nDefect Category Distribution:")
print(df["Defect_Category"].value_counts())

print("\nDescriptive Statistics:")
print(df.describe())

# --------------------------------
# 2. Defect Category Distribution
# --------------------------------

defect_counts = df["Defect_Category"].value_counts()

plt.figure(figsize=(8, 5))
defect_counts.plot(kind="bar")
plt.title("Defect Category Distribution")
plt.xlabel("Defect Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/eda_plots/defect_category_distribution.png")
plt.close()

# --------------------------------
# 3. Vibration Level Distribution
# --------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["Vibration_Level"], bins=30)
plt.title("Vibration Level Distribution")
plt.xlabel("Vibration Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/eda_plots/vibration_distribution.png")
plt.close()

# --------------------------------
# 4. Vibration Level by Defect Category
# --------------------------------

plt.figure(figsize=(8, 5))
df.boxplot(column="Vibration_Level", by="Defect_Category")
plt.title("Vibration Level by Defect Category")
plt.suptitle("")
plt.xlabel("Defect Category")
plt.ylabel("Vibration Level")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/eda_plots/vibration_by_defect.png")
plt.close()

# --------------------------------
# 5. Sensor Feature Distributions
# --------------------------------

sensor_columns = [
    "Surface_Roughness_Index",
    "Vibration_Level",
    "Temperature_Deviation",
    "Dimensional_Deviation",
    "Material_Stress_Indicator"
]

for column in sensor_columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=30)
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    filename = column.lower() + "_distribution.png"
    plt.savefig(f"data/eda_plots/{filename}")
    plt.close()

print("\nEDA completed successfully!")
print("Graphs saved in: data/eda_plots")