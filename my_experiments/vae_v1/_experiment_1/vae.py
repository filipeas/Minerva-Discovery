import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.optim import Adam

"""
Variational encoder model, used as a visual model
for our model of the world.
"""
class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 14 * 41 * 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        # x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.view(-1, 256, 14, 41)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        reconstruction = F.interpolate(reconstruction, size=(255, 701), mode="bilinear", align_corners=False)
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(14*41*256, latent_size) # 2*2*256
        self.fc_logsigma = nn.Linear(14*41*256, latent_size) # 2*2*256


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(L.LightningModule):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size, lr=1e-3):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)
        self.lr = lr

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
    
    def _loss_function(self, recon_x, x, mu, logsigma):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD
    
    def _step(self, batch, batch_idx:int, step_name:str):
        data = batch[0] # image -> torch.Size([B, C, H, W])
        # label = batch[1] # label 
        recon_batch, mu, logsigma = self.forward(data)
        loss = self._loss_function(recon_batch, data, mu, logsigma)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None):
        data = batch
        outputs = self.forward(data)
        return outputs
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == "__main__":
    print("Testing instantiate VAE")
    model = VAE(3, 512)
    print(model)