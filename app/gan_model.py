"""GAN architecture for Assignment 3: MNIST image generation."""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(z.size(0), 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.tanh(self.deconv2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.fc(x))
        return x
