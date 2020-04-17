import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, train=True, device='cpu'):
        if train:
            self.train()
        else:
            self.eval()
        x = X.to(device)
        latent_space = self.encoder(x)
        out = self.decoder(latent_space)
        return out

    def generate(self, latent_space, device='cpu'):
        self.eval()
        x = latent_space.to(device)
        return self.decoder(x)