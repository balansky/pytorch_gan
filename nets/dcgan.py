import torch.nn as nn
from nets.layers.spectral_norm import SpectralNorm


class Generator(nn.Module):

    def __init__(self, nz):
        super(Generator, self).__init__()
        self.fc = nn.Linear(nz, 7*7*64, bias=False)
        self.fc_bn = nn.BatchNorm1d(7*7*64)
        self.fc_relu = nn.ReLU()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(64, 64, 5, 1, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 5, 2, 2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.fc(input)
        x = self.fc_bn(x)
        x = self.fc_relu(x)
        x = x.view(-1, 64, 7, 7)
        x = self.main(x)
        return x


class Descriminator(nn.Module):

    def __init__(self, nc):
        super(Descriminator, self).__init__()
        self.main = nn.Sequential(
            SpectralNorm(nn.Conv2d(nc, 64, 5, 2, 16, bias=False)),
            # nn.Conv2d(nc, 64, 5, 2, 16, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            SpectralNorm(nn.Conv2d(64, 128, 5, 2, 16, bias=False)),
            # nn.Conv2d(64, 128, 5, 2, 16, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(28*28*128, 1, bias=False)
        self.a = nn.Sigmoid()

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.a(x)
        return x