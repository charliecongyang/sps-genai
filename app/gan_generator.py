"""Generate MNIST digit images using the trained GAN."""

import math
from io import BytesIO
from pathlib import Path

import torch
from torchvision.utils import save_image

from app.gan_model import Generator


class GANImageGenerator:
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "mnist_gan_generator.pt"
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                "GAN weights are missing. Run train_gan.py first."
            )

        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        self.z_dim = checkpoint["z_dim"]
        self.model = Generator(z_dim=self.z_dim).cpu()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def generate(self, num_images=16, seed=None):
        # Each request uses its own random-number generator.
        rng = torch.Generator(device="cpu")
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)

        noise = torch.randn(
            num_images, self.z_dim, generator=rng, device="cpu"
        )
        images = self.model(noise)

        buffer = BytesIO()
        save_image(
            images,
            buffer,
            format="PNG",
            nrow=math.ceil(math.sqrt(num_images)),
            normalize=True,
            value_range=(-1, 1),
        )
        return buffer.getvalue()
