import torch
from torch import nn


class SuperCov(nn.Module):
    """SuperCov Network"""

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        c1, c2, c3, c4 = 16, 16, 32, 32

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        x = self.relu(self.conv1a(data))
        x = self.relu(self.conv1b(x))
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv3a(x))

        # Compute the dense keypoint scores
        x = self.relu(self.conv4a(x))
        x = self.conv4b(x)

        return x
