"""CNN architecture specified in Assignment 2 and CIFAR-10 image inference."""

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
IMAGE_SIZE = 64


def get_transform():
    """Use the same 64 x 64 RGB preprocessing for training and inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


class AssignmentCNN(nn.Module):
    """Two convolution/pooling blocks followed by 100 and 10 output units."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Two pooling layers reduce each spatial dimension: 64 -> 32 -> 16.
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
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
        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = get_transform()

    def predict(self, image: Image.Image) -> dict:
        # Resize to the assignment's input size; ToTensor scales to [0, 1].
        inputs = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0]
        class_id = int(probabilities.argmax().item())
        return {
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "confidence": float(probabilities[class_id].item()),
        }
