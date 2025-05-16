import torch
import torch.nn as nn

class AutoencoderCF(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[256, 128]):
        super(AutoencoderCF, self).__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        encoder_layers += [nn.Linear(prev_dim, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        decoder_layers += [nn.Linear(prev_dim, input_dim)]
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed