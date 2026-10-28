"""Train the Assignment 3 GAN on MNIST."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator


ROOT = Path(__file__).resolve().parent


def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.zeros_(layer.bias)


def save_checkpoint(model, path, epoch):
    # Store weights on CPU so the Docker API can load them.
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": state,
            "epoch": epoch,
            "z_dim": 100,
            "dataset": "MNIST",
            "seed": 42,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")

    torch.manual_seed(42)

    if args.device == "auto":
        device_name = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device_name = args.device
    device = torch.device(device_name)
    print(f"Using device: {device}", flush=True)

    model_dir = ROOT / "models"
    sample_dir = ROOT / "examples" / "gan"
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Match the generator's Tanh output range: [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=str(ROOT / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"Training images: {len(dataset)}", flush=True)

    generator = Generator(z_dim=100)
    discriminator = Discriminator()
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    # Use the same noise each epoch to compare generated images.
    fixed_noise = torch.randn(64, 100, device=device)
    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_d = 0.0
        total_g = 0.0

        for batch_index, (real_images, _) in enumerate(loader, start=1):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            # Train D to distinguish real images from generated images.
            optimizer_d.zero_grad(set_to_none=True)
            loss_real = criterion(discriminator(real_images), real_labels)
            loss_fake = criterion(
                discriminator(fake_images.detach()), fake_labels
            )
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train G to make D classify generated images as real.
            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            loss_g = criterion(discriminator(fake_images), real_labels)
            loss_g.backward()
            optimizer_g.step()
            discriminator.requires_grad_(True)

            d_value = loss_d.item()
            g_value = loss_g.item()
            total_d += d_value * batch_size
            total_g += g_value * batch_size

            if batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"Batch {batch_index}/{len(loader)} "
                    f"D loss: {d_value:.4f} G loss: {g_value:.4f}",
                    flush=True,
                )

        record = {
            "epoch": epoch,
            "d_loss": total_d / len(dataset),
            "g_loss": total_g / len(dataset),
        }
        history.append(record)

        # Save a grid of 64 generated digits.
        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise).cpu()
        save_image(
            samples,
            sample_dir / f"epoch_{epoch:03d}.png",
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )

        save_checkpoint(
            generator, model_dir / "mnist_gan_generator.pt", epoch
        )
        save_checkpoint(
            discriminator, model_dir / "mnist_gan_discriminator.pt", epoch
        )

        metrics = {
            "dataset": "MNIST",
            "training_images": len(dataset),
            "epochs_completed": epoch,
            "batch_size": args.batch_size,
            "learning_rate": 0.0002,
            "seed": 42,
            "device": str(device),
            "history": history,
        }
        (model_dir / "mnist_gan_training.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch} saved. "
            f"Average D loss: {record['d_loss']:.4f}, "
            f"Average G loss: {record['g_loss']:.4f}",
            flush=True,
        )

    print("Training complete. Weights saved in models/.", flush=True)


if __name__ == "__main__":
    main()
