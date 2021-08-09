# custom weights initialization called on netG and netD
import torch
import utils.dcgan_config as ganconfig
# Create batch of latent vectors that we will use to visualize the progression of the generator

class start_seeds:
    def __init__(self, device):
        self.device = device
        self.fixed_noise = torch.randn(64, ganconfig.nz, 1, 1, device=self.device)
