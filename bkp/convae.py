import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

class ConvVAE(L.LightningModule):
    def __init__(
            self, 
            layers=[32, 64, 128, 256],
            input_channels=3, 
            z_size=32, 
            lr=0.0001):
        super(ConvVAE, self).__init__()
        self.layers = layers
        self.input_channels = input_channels
        self.z_size = z_size
        self.lr = lr

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, layers[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(layers[0], layers[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(layers[1], layers[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(layers[2], layers[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # latent space
        self.fc_mu = nn.Linear(layers[3] * 4 * 4, z_size)
        self.fc_logvar = nn.Linear(layers[3] * 4 * 4, z_size)

        # decoder
        self.fc_decode = nn.Linear(z_size, layers[3] * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(layers[3], layers[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(layers[2], layers[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(layers[1], layers[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(layers[0], self.input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        # passar x (imagem) para o encoder e retorna o mu e o logval
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # flatten para camadas lineares
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # reparametriza a amostra pro espaço latente z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # passa o espaço latente (z) para o decoder
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.layers[3], 4, 4) # reshape para convolucoes transpostas
        return self.decoder(h)
    
    def forward(self, x):
        # forwar pass do VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # calcula a loss combinada
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss
    
    def _step(self, batch, step_name):
        x, _ = batch # o batch vem a imagem e a label
        recon_x, mu, logvar = self(x) # chama o forward
        loss = self.loss_function(recon_x, x, mu, logvar)
        self.log(f"{step_name}_loss", loss, 
                 on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        
        # Log de estatísticas de mu e logvar
        self.log(f"{step_name}_mu_mean", mu.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{step_name}_mu_std", mu.std(), on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{step_name}_logvar_mean", logvar.mean(), on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{step_name}_logvar_std", logvar.std(), on_epoch=True, prog_bar=False, logger=True)

        if step_name == 'val':
            # Armazenar mu e logvar para análise
            if not hasattr(self, "logged_mu"):
                self.logged_mu = []
                self.logged_logvar = []
            self.logged_mu.append(mu.detach().cpu())
            self.logged_logvar.append(logvar.detach().cpu())

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, step_name="train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, step_name="val")
    
    def on_validation_epoch_end(self):
        if hasattr(self, "logged_mu") and hasattr(self, "logged_logvar"):
            # Concatenar os valores acumulados
            mu_values = torch.cat(self.logged_mu).numpy()
            logvar_values = torch.cat(self.logged_logvar).numpy()

            # Salvar em arquivos para análise posterior
            np.save("mu_values.npy", mu_values)
            np.save("logvar_values.npy", logvar_values)

            # Limpar os valores para a próxima época
            self.logged_mu = []
            self.logged_logvar = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)