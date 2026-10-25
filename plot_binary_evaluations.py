"""Plot real binary checkpoint validation predictions into report artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_curve, roc_auc_score)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "experiments" / "binary_baselines"


def load_experiment(modality: str) -> dict:
    evaluation_path = ROOT / "artifacts" / "logs" / f"{modality}_binary_10k_v2_evaluation.json"
    predictions_path = ROOT / "artifacts" / "logs" / f"{modality}_binary_10k_v2_validation_predictions.csv"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metrics = evaluation["metrics"]
    positive_label = metrics["positive_class"]
    labels = evaluation["class_names"]
    y_true = np.asarray([row["actual_label"] == positive_label for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["positive_probability"]) for row in rows], dtype=np.float64)
    return {"modality": modality, "evaluation": evaluation, "metrics": metrics,
            "labels": labels, "positive": positive_label, "y_true": y_true,
            "scores": scores, "sample_ids": [row["sample_id"] for row in rows]}


def save_figure(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_epoch_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, modality, title, color in (
        (axes[0], "xray", "X-ray / DenseNet-121", "#2D6CDF"),
        (axes[1], "skin", "Skin / EfficientNet-B4", "#0F9D91"),
    ):
        log_path = ROOT / "artifacts" / "logs" / f"{modality}_binary_10k_v2_training.jsonl"
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if "epoch" in row]
        epochs = [row["epoch"] for row in rows]
        train_loss = [row["train_loss"] for row in rows]
        val_loss = [row["validation_loss"] for row in rows]
        train_accuracy = [row["train_metrics"]["accuracy"] for row in rows]
        val_accuracy = [row["validation_metrics"]["accuracy"] for row in rows]
        auc = [row["validation_metrics"]["roc_auc_positive_class"] for row in rows]
        sensitivity = [row["validation_metrics"]["sensitivity"] for row in rows]
        ax.plot(epochs, train_loss, marker="o", color=color, label="Train loss")
        ax.plot(epochs, val_loss, marker="s", linestyle="--", color="#C05A00", label="Validation loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_xticks(epochs); ax.grid(alpha=.2); ax.set_title(title)
        right = ax.twinx()
        right.plot(epochs, auc, marker="^", color="#6E59A5", label="Validation ROC-AUC")
        right.plot(epochs, sensitivity, marker="D", color="#14805E", label="Validation sensitivity")
        right.plot(epochs, train_accuracy, linestyle=":", color="#7D8793", label="Train accuracy")
        right.plot(epochs, val_accuracy, linestyle="-.", color="#D24A5A", label="Validation accuracy")
        right.set_ylabel("Rate / score"); right.set_ylim(0, 1.02)
        lines, labs = ax.get_legend_handles_labels(); lines2, labs2 = right.get_legend_handles_labels()
        ax.legend(lines + lines2, labs + labs2, fontsize=7, frameon=False, loc="best")
    fig.suptitle("Recorded training and validation learning curves", fontweight="bold")
    save_figure(fig, "epoch_learning_curves")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = [load_experiment("xray"), load_experiment("skin")]
    colors = {"xray": "#2D6CDF", "skin": "#0F9D91"}
    make_epoch_curves()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, item, label in zip(axes, data, ("Pneumonia", "MEL")):
        fpr, tpr, _ = roc_curve(item["y_true"], item["scores"])
        auc = roc_auc_score(item["y_true"], item["scores"])
        ax.plot(fpr, tpr, color=colors[item["modality"]], lw=2, label=f"{label} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#8A96A3", lw=1, label="Chance")
        ax.set(xlim=(0,1), ylim=(0,1.02), xlabel="False positive rate", ylabel="Sensitivity / true positive rate")
        ax.set_title(f"{item['modality'].upper()} validation ROC")
        ax.grid(alpha=.2); ax.legend(frameon=False, loc="lower right")
    fig.suptitle("ROC curves from saved validation predictions", fontweight="bold")
    save_figure(fig, "validation_roc_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, item, label in zip(axes, data, ("Pneumonia", "MEL")):
        precision, recall, _ = precision_recall_curve(item["y_true"], item["scores"])
        ap = average_precision_score(item["y_true"], item["scores"])
        prevalence = item["y_true"].mean()
        ax.plot(recall, precision, color=colors[item["modality"]], lw=2, label=f"{label} (AP={ap:.3f})")
        ax.axhline(prevalence, color="#8A96A3", linestyle="--", lw=1, label=f"Prevalence={prevalence:.3f}")
        ax.set(xlim=(0,1), ylim=(0,1.02), xlabel="Recall / sensitivity", ylabel="Precision")
        ax.set_title(f"{item['modality'].upper()} validation precision-recall")
        ax.grid(alpha=.2); ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Precision-recall curves from saved validation predictions", fontweight="bold")
    save_figure(fig, "validation_precision_recall_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, item, label in zip(axes, data, ("Pneumonia", "MEL")):
        thresholds = np.linspace(0, 1, 1001)
        sens, spec = [], []
        for threshold in thresholds:
            predicted = item["scores"] >= threshold
            positives = item["y_true"] == 1
            negatives = ~positives
            sens.append(float((predicted & positives).sum() / max(positives.sum(), 1)))
            spec.append(float((~predicted & negatives).sum() / max(negatives.sum(), 1)))
        chosen = item["metrics"]["decision_threshold"]
        ax.plot(thresholds, sens, color="#C84C09", label="Sensitivity")
        ax.plot(thresholds, spec, color="#2D6CDF", label="Specificity")
        ax.axvline(chosen, color="#25364A", linestyle="--", label=f"Saved validation threshold={chosen:.3f}")
        ax.set(xlim=(0,1), ylim=(0,1.02), xlabel="Positive-class probability threshold", ylabel="Rate")
        ax.set_title(f"{item['modality'].upper()} operating point")
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sensitivity-specificity tradeoff; thresholds use validation data", fontweight="bold")
    save_figure(fig, "validation_threshold_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, item, label in zip(axes, data, ("Pneumonia", "MEL")):
        observed, predicted = calibration_curve(item["y_true"], item["scores"], n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, marker="o", color=colors[item["modality"]], label=label)
        ax.plot([0,1], [0,1], "--", color="#8A96A3", lw=1, label="Ideal")
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean predicted probability", ylabel="Observed positive fraction")
        ax.set_title(f"{item['modality'].upper()} calibration (10 bins)")
        ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("Calibration curves from validation predictions (no recalibration fitted)", fontweight="bold")
    save_figure(fig, "validation_calibration_curves")

    summary_rows = []
    for item in data:
        m = item["metrics"]
        ci = item["evaluation"].get("confidence_intervals_95", {})
        summary_rows.append({
            "modality": item["modality"], "positive_class": item["positive"],
            "validation_count": item["evaluation"]["validation_count"],
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "roc_auc": m["roc_auc_positive_class"], "average_precision": m["average_precision_positive_class"],
            "sensitivity": m["sensitivity"], "sensitivity_ci_low": ci.get("sensitivity", [None,None])[0],
            "sensitivity_ci_high": ci.get("sensitivity", [None,None])[1],
            "specificity": m["specificity"], "specificity_ci_low": ci.get("specificity", [None,None])[0],
            "specificity_ci_high": ci.get("specificity", [None,None])[1],
            "positive_precision": m["positive_precision"], "positive_f1": m["positive_f1"],
            "mcc": m["matthews_correlation_coefficient"], "brier_score": m["brier_score_positive_class"],
            "ece_10_bins": m["expected_calibration_error_10_bins"],
            "threshold": m["decision_threshold"], "confusion_matrix": json.dumps(m["confusion_matrix"]),
            "validation_note": item["evaluation"]["validation_note"],
        })
    columns = list(summary_rows[0])
    with (OUT / "binary_validation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(summary_rows)
    (OUT / "binary_validation_metrics.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Wrote validation-only figures and metric tables to {OUT}")


if __name__ == "__main__":
    main()
