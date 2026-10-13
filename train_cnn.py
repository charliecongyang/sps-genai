"""Train the CNN specified in Assignment 2 on CIFAR-10 and save it for the API."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from app.cnn_model import (
    CLASSES, IMAGE_SIZE, MODEL_PATH, AssignmentCNN, ImageClassifier, get_transform,
)


ROOT = Path(__file__).resolve().parent


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, data_loader, criterion, optimizer, device, epochs):
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_accuracy": correct / total,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}/{epochs}: loss={row['train_loss']:.4f}, "
            f"accuracy={row['train_accuracy']:.2%}, time={row['seconds']:.1f}s",
            flush=True,
        )
    return history


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch-size and threads must be positive")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    # CIFAR-10 is 32 x 32; the assignment requires resizing it to 64 x 64.
    transform = get_transform()
    train_data = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    history = train_model(model, train_loader, criterion, optimizer, device, args.epochs)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test accuracy on {len(test_data)} images: {test_accuracy:.2%}", flush=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
        "classes": list(CLASSES),
        "architecture": "AssignmentCNN",
        "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "epochs": args.epochs,
        "test_accuracy": test_accuracy,
    }, MODEL_PATH)

    example_dir = ROOT / "examples"
    example_dir.mkdir(exist_ok=True)
    classifier = ImageClassifier()
    examples = []
    for index, filename in [(0, "test_image.png"), (1, "test_ship.png")]:
        # Save the original image, then use the exact API preprocessing path.
        image = Image.fromarray(test_data.data[index])
        image.save(example_dir / filename)
        examples.append({
            "file": filename, "test_index": index,
            "true_class": CLASSES[test_data.targets[index]],
            **classifier.predict(image),
        })
    (example_dir / "predictions.json").write_text(json.dumps(examples, indent=2) + "\n")
    metrics = {
        "dataset": "CIFAR-10",
        "model": "AssignmentCNN",
        "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "preprocessing": "RGB, Resize(64, 64), ToTensor (pixels in [0, 1])",
        "seed": 42,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.0005,
        "train_examples": len(train_data),
        "test_examples": len(test_data),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "example_true_class": CLASSES[test_data.targets[0]],
    }
    (MODEL_PATH.parent / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved weights: {MODEL_PATH}", flush=True)
    print(f"Example predictions: {example_dir / 'predictions.json'}", flush=True)


if __name__ == "__main__":
    main()
