import torch
from torch import nn
import torch.nn.functional as F
from models import GLinear, GConv2d

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, mode):
        super().__init__()
        self.b = nn.Sequential(
            GConv2d(input_feature=in_channels, output_feature=out_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, bias=False, mode=mode),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.b(x)


class ResNet(nn.Module):
    def __init__(self, mode, class_num):
        super().__init__()
        self.b1 = nn.Sequential(
            Conv(mode=mode,
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            Conv(mode=mode,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.b3 = nn.Sequential(
            Conv(mode=mode,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.b4 = nn.Sequential(
            Conv(mode=mode,
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.b4_x = Conv(mode=mode,
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.b5 = nn.Sequential(
            Conv(mode=mode,
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.b6 = nn.Sequential(
            Conv(mode=mode,
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.b6_x = Conv(mode=mode,
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.b7 = nn.Sequential(
            Conv(mode=mode,
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.b8 = nn.Sequential(
            Conv(mode=mode,
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.b8_x = Conv(mode=mode,
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.b9 = nn.Sequential(
            Conv(mode=mode,
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv(mode=mode,
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.b10 = GLinear(input_size=512, output_size=class_num, bias=True)

    def forward(self, x):
        x = self.b1(x)
        x = F.relu(self.b2(x) + x)
        x = F.relu(self.b3(x) + x)
        x = F.relu(self.b4(x) + self.b4_x(x))
        x = F.relu(self.b5(x) + x)
        x = F.relu(self.b6(x) + self.b6_x(x))
        x = F.relu(self.b7(x) + x)
        x = F.relu(self.b8(x) + self.b8_x(x))
        x = F.relu(self.b9(x) + x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.b10(x)
        return x