import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class KanjiNet(nn.Module):
    """A 5x5 stride-2 stem takes 64 px to 32 px at once, then stages at 32, 16, 8 and 4 px,
    global average pool, dropout, and one linear head per task.

    head_sizes: {"character": 3036, "nested": 382, "positioned": 251}; only "character" is required.
    """

    def __init__(self, head_sizes, width=16, dropout=0.2):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 8]
        self.features = nn.Sequential(
            conv_block(1, channels[0], kernel_size=5, stride=2),
            conv_block(channels[0], channels[0]),
            nn.MaxPool2d(2),
            conv_block(channels[0], channels[1]),
            conv_block(channels[1], channels[1]),
            nn.MaxPool2d(2),
            conv_block(channels[1], channels[2]),
            conv_block(channels[2], channels[2]),
            nn.MaxPool2d(2),
            conv_block(channels[2], channels[3]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.embedding_size = channels[3]
        self.heads = nn.ModuleDict({name: nn.Linear(channels[3], size) for name, size in head_sizes.items()})

    def forward(self, images):
        embedding = self.features(images)
        return {name: head(embedding) for name, head in self.heads.items()}


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
