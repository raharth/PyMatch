import torch.nn as nn
from sklearn.neighbors import KernelDensity


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

    def expand(self, latent_space, device='cpu'):
        self.eval()
        x = latent_space.to(device)
        return self.decoder(x)


class DeepGenerator(AutoEncoder):

    def __init__(self, encoder, decoder):
        super(DeepGenerator, self).__init__(encoder, decoder)
        self.kde = None

    def estimate_latent_density(self, data):
        latent = self.encoder(data)
        self.kde = KernelDensity(bandwidth=.1, kernel='gaussian')
        self.kde.fit(latent.detach().numpy())

    def sample(self, n_samples: int, device='cpu'):
        latent = self.kde.sample(n_samples=n_samples)
        return self.expand(latent, device=device)
