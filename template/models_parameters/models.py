import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class SRCNN(nn.Module):
    # https://keras.io/examples/vision/super_resolution_sub_pixel/
    # https://mfarahmand.medium.com/cnn-based-single-image-super-resolution-6ffcd39ec993
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x