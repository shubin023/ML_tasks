import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5, linear_hidden_size=256):
        super().__init__()

        self.conv_start = nn.Conv2d(3, start_channels, 1, 1, 0)

        in_num = start_channels
        self.downsample_blocks = nn.ModuleList()
        for _ in range(downsamplings):
            self.downsample_blocks.append(nn.Sequential(
                                              nn.Conv2d(in_num, in_num * 2, 3, 2, 1),
                                              nn.BatchNorm2d(in_num * 2),
                                              nn.ReLU()
                                          ))
            in_num *= 2
        
        self.flatten_layer = nn.Flatten(start_dim=1)

        self.linear_model = nn.Sequential(
                                nn.Linear(in_features=img_size ** 2 // (2 ** downsamplings) * start_channels, out_features=linear_hidden_size),
                                nn.ReLU(),
                                nn.Linear(in_features=linear_hidden_size, out_features=2 * latent_size)
                            )

    def forward(self, x):
        x = self.conv_start(x)
        for layer in self.downsample_blocks:
            x = layer(x)
        x = self.flatten_layer(x)
        x = self.linear_model(x)

        mu = x[:, :x.shape[1] // 2]
        sigma = torch.exp(x[:, x.shape[1] // 2:])
        e_rand = torch.randn(x.shape[0], x.shape[1] // 2).to(x.device)
        z = mu + e_rand * sigma
        
        return z, (mu, sigma)
    
    
# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5, linear_hidden_size=256):
        super().__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.end_channels = end_channels
        self.upsamplings = upsamplings
        self.linear_hidden_size = linear_hidden_size

        self.linear_model = nn.Sequential(
                                nn.Linear(in_features=latent_size, out_features=linear_hidden_size * 2),
                                nn.ReLU(),
                                nn.Linear(in_features=linear_hidden_size * 2, out_features=img_size ** 2 // (2 ** upsamplings) * end_channels)
                              )

        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=(end_channels * (2 ** upsamplings), img_size // (2 ** upsamplings), img_size // (2 ** upsamplings)))

        in_num = end_channels * (2 ** upsamplings)
        self.upsampling_blocks = nn.ModuleList()
        for _ in range(upsamplings):
            self.upsampling_blocks.append(nn.Sequential(
                                              nn.ConvTranspose2d(in_num, out_channels=in_num // 2, kernel_size=4, stride=2, padding=1),
                                              nn.BatchNorm2d(in_num // 2),
                                              nn.ReLU()
                                            ))
            in_num //= 2

        self.conv_tanh = nn.Sequential(
                            nn.Conv2d(self.end_channels, 3, 1, 1, 0),
                            nn.Tanh()
                          )

    def forward(self, z):
        z = self.linear_model(z)
        z = self.unflatten_layer(z)
        for layer in self.upsampling_blocks:
            z = layer(z)
        x_pred = self.conv_tanh(z)
        return x_pred
    
# Task 3


class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=3, latent_size=256, down_channels=6, up_channels=12, linear_hidden_size=178):
        super().__init__()
        self.encoder = Encoder(img_size, latent_size, down_channels, downsamplings, linear_hidden_size)
        self.decoder = Decoder(img_size, latent_size, up_channels, downsamplings, linear_hidden_size)
        
    def forward(self, x):
        z, mu_sigma = self.encoder(x)
        mu, sigma = mu_sigma[0], mu_sigma[1]
        kld = 0.5 * (sigma ** 2 + mu ** 2 - torch.log(sigma ** 2) - 1)
        x_pred = self.decoder(z)
        return x_pred, kld

    def encode(self, x):
        z, _ = self.encoder(x)
        return z
    
    def decode(self, z):
        x_pred = self.decoder(z)
        return x_pred
    
    def save(self):
        model_path = __file__[:-7] + "VAE_model.pth"
        torch.save(self.state_dict(), model_path)
    
    def load(self):
        model_path = __file__[:-7] + "VAE_model.pth"
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))