import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """3 x [conv128 -> GroupNorm -> ReLU -> AvgPool] + linear (identical to the official utils.py)."""

    def __init__(self, channel=3, net_width=128, net_depth=3, net_norm="groupnorm",
                 net_pooling="avgpooling", num_classes=10, im_size=(32, 32)):
        super().__init__()
        layers, shape = [], [channel, im_size[0], im_size[1]]
        in_ch = channel
        for _ in range(net_depth):
            layers += [nn.Conv2d(in_ch, net_width, kernel_size=3, padding=1)]
            shape[0] = net_width
            layers += [nn.GroupNorm(4, net_width, affine=True), nn.ReLU(inplace=True)]
            in_ch = net_width
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            shape[1] //= 2
            shape[2] //= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(shape[0] * shape[1] * shape[2], num_classes)

    def embed(self, x):
        return self.features(x).view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self.embed(x))


class LSTMClassifier(nn.Module):
    """2-layer bidirectional LSTM classifier used on the AWE data (official main_awe/utils.py)."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def embed(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1, :])

    def forward(self, x):
        return self.fc(self.embed(x))


class LSTMVAE(nn.Module):
    """Lightweight LSTM-VAE used as the generative proxy on time series (official main_awe/utils.py)."""

    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.input_dim, self.hidden_dim, self.latent_dim = input_dim, hidden_dim, latent_dim
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, seq_len):
        h0 = torch.tanh(self.fc_dec(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_in = torch.zeros(z.size(0), seq_len, self.hidden_dim, device=z.device)
        out, _ = self.decoder_lstm(dec_in, (h0, c0))
        return self.output_layer(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar), x.size(1)), mu, logvar


def vae_loss(x, x_recon, mu, logvar):
    recon = torch.nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 0.1 * kl
