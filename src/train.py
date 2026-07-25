"""
Training script for the Pneumonia Detection model.

Usage:
    python -m src.train --epochs 15 --batch-size 32 --lr 1e-4

Produces:
    models/pneumonia_densenet121.pt   (best checkpoint, by validation accuracy)
    reports/training_history.json     (loss/accuracy per epoch, used for plots)
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    MODELS_DIR, MODEL_PATH, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    RANDOM_SEED, DEVICE, BASE_DIR,
)
from src.dataset import get_dataloaders
from src.model import build_model


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return {"loss": avg_loss, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Freeze convolutional backbone, train classifier head only "
                              "(faster, good first pass on CPU)")
    args = parser.parse_args()

    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=args.batch_size)
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset)} | "
          f"Test samples: {len(test_loader.dataset)}")

    # Handle class imbalance (pneumonia dataset typically has ~3x more
    # PNEUMONIA images than NORMAL) using class weights in the loss.
    targets = [label for _, label in train_loader.dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = torch.tensor(
        [len(targets) / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"Class counts: {class_counts.tolist()} | Class weights: {class_weights.tolist()}")

    model = build_model(pretrained=True, freeze_backbone=args.freeze_backbone).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = run_epoch(model, val_loader, criterion, None, DEVICE)
        scheduler.step(val_metrics["accuracy"])
        elapsed = time.time() - t0

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['f1']:.4f} ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "epoch": epoch,
                "classes": classes,
            }, MODEL_PATH)
            print(f"  -> Saved new best checkpoint (val_acc={best_val_acc:.4f}) to {MODEL_PATH}")

    # Final test-set evaluation using the best checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, None, DEVICE)
    print(f"\nFINAL TEST METRICS: {test_metrics}")

    history.append({"final_test_metrics": test_metrics})
    with open(BASE_DIR / "reports" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
