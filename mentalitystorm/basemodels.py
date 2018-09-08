import torch
from torch import nn as nn

from mentalitystorm import Dispatcher, Observable, Trainable, TensorBoardObservable


class BaseVAE(nn.Module, Dispatcher, Observable, Trainable, TensorBoardObservable):
    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)
        Dispatcher.__init__(self)
        TensorBoardObservable.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, noise=True):
        input_shape = x.shape
        indices = None

        self.updateObserversWithImage('input', x[0], training=self.training)

        encoded = self.encoder(x)
        mu = encoded[0]
        logvar = encoded[1]
        if len(encoded) > 2:
            indices = encoded[2]

        # if z can be shown as an image dispatch it
        if mu.shape[1] == 3 or mu.shape[1] == 1:
            self.updateObserversWithImage('z', mu[0].data)
        self.metadata['z_size'] = mu[0].data.numel()

        z = self.reparameterize(mu, logvar, noise=noise)

        if indices is not None:
            decoded = self.decoder(z, indices)
        else:
            decoded = self.decoder(z)

        # should probably make decoder return same shape as encoder
        decoded = decoded.view(input_shape)

        self.updateObserversWithImage('output', decoded[0].data, training=self.training)

        return decoded, mu, logvar

    def reparameterize(self, mu, logvar, noise=True):
        if self.training and noise:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
