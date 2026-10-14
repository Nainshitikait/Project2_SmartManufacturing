import pandas as pd
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/manufacturing_sensor_data.csv"

EVALUATION_DIR = "data/model_evaluation"

CONFUSION_DIR = os.path.join(
    EVALUATION_DIR,
    "confusion_matrices"
)


# ============================================================
# CREATE EVALUATION FOLDERS
# ============================================================

os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n==============================================")
print("SMART MANUFACTURING - MODEL EVALUATION")
print("==============================================")

print(f"\nDataset shape: {df.shape}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

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


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        random_state=42
    ),

    "SVM": Pipeline([
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
}


# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

results = []

classification_reports = []

class_labels = sorted(y.unique())


for model_name, model in models.items():

    print("\n----------------------------------------------")
    print(f"Training: {model_name}")
    print("----------------------------------------------")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    predictions = model.predict(
        X_test
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (sec)": training_time
    })


    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Training Time  : {training_time:.2f} sec")


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    classification_reports.append(
        f"\n{'=' * 60}\n"
        f"{model_name}\n"
        f"{'=' * 60}\n"
        f"{report}\n"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    ax.set_title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    confusion_path = os.path.join(
        CONFUSION_DIR,
        f"{safe_name}.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Confusion matrix saved: {confusion_path}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("\n\n==============================================")
print("FINAL MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "Training Time (sec)": "{:.2f}".format
        }
    )
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_path = (
    "data/model_comparison_results.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nModel comparison saved to:"
    f"\n{comparison_path}"
)


# ============================================================
# SAVE CLASSIFICATION REPORTS
# ============================================================

reports_path = os.path.join(
    EVALUATION_DIR,
    "classification_reports.txt"
)

with open(
    reports_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SMART MANUFACTURING - CLASSIFICATION REPORTS\n"
    )

    file.write(
        "Dataset: manufacturing_sensor_data.csv\n"
    )

    file.write(
        "Test Size: 20%\n"
    )

    file.write(
        "Random State: 42\n\n"
    )

    for report in classification_reports:

        file.write(report)


print(
    f"Classification reports saved to:"
    f"\n{reports_path}"
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n==============================================")
print("BEST MODEL")
print("==============================================")

print(
    f"Model     : {best_model['Model']}"
)

print(
    f"Accuracy  : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision : {best_model['Precision']:.4f}"
)

print(
    f"Recall    : {best_model['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_model['F1 Score']:.4f}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("EVALUATION COMPLETE")
print("==============================================")