from torch.nn import modules as nn
import torch.nn.functional as F
from mentalitystorm import BaseVAE, Storeable


class MaxPooling(Storeable, BaseVAE):
    def __init__(self):
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)

    class Encoder(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.cn1 = nn.Conv2d(3, 3, kernel_size=3, stride=1)
            self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        def forward(self, x):

            z, index = self.mp1(F.relu(self.cn1(x)))
            return z, None, index

    class Decoder(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.ct1 = nn.ConvTranspose2d(3, 3, kernel_size=3)
            self.mup1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        def forward(self, z, index):
            decoded = self.mup1(z, index)
            return F.relu(self.ct1(decoded))


class ConvolutionalPooling(Storeable, BaseVAE):
    def __init__(self):
        encoder = self.Encoder()
        decoder = self.Decoder()
        BaseVAE.__init__(self, encoder, decoder)
        Storeable.__init__(self)

    class Encoder(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.cn1 = nn.Conv2d(3, 3, kernel_size=3, stride=1)
            self.mp1 = nn.Conv2d(3, 3, kernel_size=2, stride=2)

        def forward(self, x):

            z = self.mp1(F.relu(self.cn1(x)))
            return z, None

    class Decoder(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.ct1 = nn.ConvTranspose2d(3, 3, kernel_size=3)
            self.mup1 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)

        def forward(self, z):
            decoded = self.mup1(z)
            return F.relu(self.ct1(decoded))
