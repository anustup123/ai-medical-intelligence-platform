"""
Standalone evaluation script — produces a confusion matrix image and a
classification report for the test set, using an already-trained checkpoint.

Usage:
    python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

from src.config import MODEL_PATH, DEVICE, BASE_DIR, LABELS
from src.dataset import get_dataloaders
from src.model import load_trained_model


def evaluate():
    _, _, test_loader, classes = get_dataloaders()
    model = load_trained_model(str(MODEL_PATH), device=DEVICE)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(PNEUMONIA)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=LABELS, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=LABELS))

    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC: {auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = BASE_DIR / "reports" / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve — Test Set"); ax2.legend()
    fig2.tight_layout()
    roc_path = BASE_DIR / "reports" / "roc_curve.png"
    fig2.savefig(roc_path, dpi=150)
    print(f"Saved ROC curve to {roc_path}")

    with open(BASE_DIR / "reports" / "test_classification_report.json", "w") as f:
        json.dump({"report": report, "roc_auc": auc}, f, indent=2)


if __name__ == "__main__":
    evaluate()
