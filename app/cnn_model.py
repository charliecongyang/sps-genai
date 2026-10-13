"""CIFAR-10 classifier adapted from Module 4 Practical 3, EnhancedCNN."""

from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.pt"


class EnhancedCNN(nn.Module):
    """The four-convolution architecture used in the classroom practical."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ImageClassifier:
    """Load trained weights once and classify uploaded images on the CPU."""

    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                "CNN weights are missing. Run: uv run python train_cnn.py"
            )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("The saved class order does not match CIFAR-10.")
        self.model = EnhancedCNN()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image) -> dict:
        # ToTensor scales pixels to [0, 1], matching the training practical.
        inputs = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0]
        class_id = int(probabilities.argmax().item())
        return {
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "confidence": float(probabilities[class_id].item()),
        }
