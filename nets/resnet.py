import torch
import math
from .layers.categorical_batch_norm import CategoricalBatchNorm


class Gblock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, num_categories=None,
                 kernel_size=3, stride=1, padding=1, upsample=True):
        super(Gblock, self).__init__()
        hidden_channels = out_channels if not hidden_channels else hidden_channels
        self.num_categories = num_categories
        self.upsample = upsample

        self.conv1 = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.s_conv = None
        torch.nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if in_channels != out_channels:
            self.s_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            torch.nn.init.xavier_uniform_(self.s_conv.weight.data, 1.)

        self.bn1 = self.batch_norm(in_channels)
        self.bn2 = self.batch_norm(hidden_channels)
        self.activate = torch.nn.ReLU()
        self.up = lambda a: torch.nn.functional.interpolate(a, scale_factor=2)

    def batch_norm(self, num_features):
        return torch.nn.BatchNorm2d(num_features) if not self.num_categories \
            else CategoricalBatchNorm(num_features, self.num_categories)

    def forward(self, input, y=None):
        x_r = input
        x = self.bn1(input, y) if self.num_categories else self.bn1(input)
        x = self.activate(x)
        if self.upsample:
            x = self.up(x)
            x_r = self.up(x_r)
        x = self.conv1(x)
        x = self.bn2(x, y) if self.num_categories else self.bn2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.s_conv:
            x_r = self.s_conv(x_r)
        return x + x_r


class ResnetGenerator(torch.nn.Module):

    def __init__(self, ch=64, z_dim=128, n_categories=None, bottom_width=4):
        super(ResnetGenerator, self).__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.n_categories = n_categories
        self.bottom_width = bottom_width
        self.dense = torch.nn.Linear(self.z_dim, bottom_width * bottom_width * ch * 16)
        torch.nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        self.block2 = Gblock(ch*16, ch*16, upsample=True)
        self.block3 = Gblock(ch * 16, ch * 8, upsample=True)
        self.block4 = Gblock(ch * 8, ch * 4, upsample=True)
        self.block5 = Gblock(ch * 4, ch * 2, upsample=True)
        self.block6 = Gblock(ch * 2, ch, upsample=True)
        self.final = self.final_block()


    def final_block(self):
        conv = torch.nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv.weight.data, 1.)
        final_ = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.ch),
            torch.nn.ReLU(),
            conv,
            torch.nn.Tanh()
        )
        return final_


    def forward(self, input, y=None):
        x = self.dense(input)
        x = x.view(-1, self.ch*16, self.bottom_width, self.bottom_width)
        x = self.block2(x, y)
        x = self.block3(x, y)
        x = self.block4(x, y)
        x = self.block5(x, y)
        x = self.block6(x, y)
        x = self.final(x)
        return x

