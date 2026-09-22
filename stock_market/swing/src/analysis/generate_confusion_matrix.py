"""
Confusion Matrix & Classification Metrics Generator for Streamlined 6-Class ML Selector Model

Calculates true vs predicted classes across expanding-window walk-forward predictions (2010 to 2026).
Plots heatmaps and exports precision, recall, F1-score evaluation metrics.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.run_streamlined_multi_class_benchmark import prepare_streamlined_6class_dataset, run_walk_forward_6class_model

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def generate_6class_confusion_matrix():
    print("=========================================================================")
    print("GENERATING ACCURATE CONFUSION MATRIX & EVALUATION METRICS FOR 6-CLASS ML")
    print("=========================================================================\n", flush=True)

    df_6c = prepare_streamlined_6class_dataset()

    df_6c["C2_Date"] = pd.to_datetime(df_6c["C2_Date"])
    df_6c["Year"] = df_6c["C2_Date"].dt.year
    df_sc1 = df_6c[df_6c["Scenario_1to2"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    # Get Walk-Forward ML Predictions
    df_pred_all = run_walk_forward_6class_model(df_sc1, probability_threshold=0.42)

    # Map true labels & predicted labels
    y_true = df_pred_all["Streamlined_6Class_Label"].values

    y_pred = []
    for idx, row in df_pred_all.iterrows():
        rr_choice = row["ML_RR_Choice"]
        if rr_choice == "1:3":
            y_pred.append(2)
        elif rr_choice == "1:2":
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_pred = np.array(y_pred)

    class_names = ["SkipTrade (0)", "Target 1:2 (1)", "Target 1:3+ (2)"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0

    print("--- RAW CONFUSION MATRIX (COUNTS) ---")
    print(pd.DataFrame(cm, index=[f"True: {c}" for c in class_names], columns=[f"Pred: {c}" for c in class_names]).to_string(), flush=True)

    print("\n--- NORMALIZED CONFUSION MATRIX (%) ---")
    print(pd.DataFrame(cm_norm, index=[f"True: {c}" for c in class_names], columns=[f"Pred: {c}" for c in class_names]).to_string(), flush=True)

    # Classification Report
    crep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_crep = pd.DataFrame(crep).transpose()
    print("\n--- CLASSIFICATION METRICS REPORT ---")
    print(df_crep.to_string(), flush=True)

    out_csv = REPORTS_DIR / "Confusion_Matrix_Classification_Report_6Class.csv"
    df_crep.to_csv(out_csv)

    # Plot Confusion Matrix Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=300)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Confusion Matrix (Trade Counts)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Class", fontsize=10, fontweight="bold")

    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Greens", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title("Normalized Confusion Matrix (% Recall per True Class)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax2.set_ylabel("True Class", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrix Evaluation: Streamlined 6-Class ML Selector Model", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_png = BASE_DIR / "Confusion_Matrix_6Class_ML_Selector.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"\nConfusion Matrix Heatmap saved to: {out_png.resolve()}", flush=True)
    return str(out_png)

if __name__ == "__main__":
    generate_6class_confusion_matrix()
