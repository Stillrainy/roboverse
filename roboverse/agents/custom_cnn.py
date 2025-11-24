import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for SB3 that takes (4,H,W) input (RGB + Depth).
    No normalization is done inside this class — expect pre-normalized inputs.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]   # should be 4

        # --- CNN Backbone ---
        # Similar to a small ResNet-style stem + 3 blocks

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Determine the size of CNN output dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).view(1, -1).shape[1]

        # Final MLP head (compress features to desired dim)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True)
        )

        self._features_dim = features_dim

    def forward(self, x):
        # x: [batch, 4, H, W]
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
