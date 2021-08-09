import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision.utils import save_image
from utils.img_Dataloader import srgan_dataloader
import utils.srgan_config as ganconfig
from mymodels.Gan_structure import SRGAN_Generator,SRGAN_Discriminator,weights_init


if __name__ == '__main__':

    dataloader = srgan_dataloader()
    dataloader_LR,dataloader_HR = dataloader.loader()

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ganconfig.ngpu > 0) else "cpu")

    # Create the generator
    netG = SRGAN_Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the Discriminator
    netD = SRGAN_Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    # Initialize MSELoss function
    loss = nn.MSELoss()
    # loss = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=ganconfig.lr, betas=(ganconfig.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=ganconfig.lr, betas=(ganconfig.beta1, 0.999))

    Tensor = torch.cuda.FloatTensor
    for epoch in range(ganconfig.num_epochs):
        for i, ((imgs_HR, _), (imgs_LR, _)) in enumerate(zip(dataloader_HR, dataloader_LR)):

            batch_size_HR = imgs_HR.shape[0]
            batch_size_LR = imgs_LR.shape[0]
            # Adversarial ground truths
            valid = Variable(Tensor(batch_size_HR, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch_size_LR, 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs_HR = Variable(imgs_HR.type(Tensor).expand(imgs_HR.size(0), 3, ganconfig.image_size_HR, ganconfig.image_size_HR))
            imgs_LR = Variable(imgs_LR.type(Tensor).expand(imgs_LR.size(0), 3, ganconfig.image_size_LR, ganconfig.image_size_LR))

            # ------------------
            #  Train Generators
            # ------------------

            optimizerG.zero_grad()

            # Generate a batch of images
            gen_imgs = netG(imgs_LR)
            # Determine validity of generated images
            validity = netD(gen_imgs)

            g_loss = (loss(gen_imgs, imgs_HR))

            g_loss.backward()
            optimizerG.step()

            # ----------------------
            #  Train Discriminators
            # ----------------------

            optimizerD.zero_grad()

            # Determine validity of real and generated images
            validity_real = netD(imgs_HR)
            validity_fake = netD(gen_imgs.detach())

            d_loss = (loss(validity_real, valid) + loss(validity_fake, fake)) / 2

            d_loss.backward()

            optimizerD.step()

            batches_done = epoch * len(dataloader_HR) + i
            if batches_done % ganconfig.sample_interval == 0:
                gen_imgs = torch.cat((imgs_HR.data, gen_imgs.data), 0)

                dirs = "./resources/srgan_compare/%s/" % (ganconfig.datasetName)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                save_image(gen_imgs, dirs + "%d.png"%(batches_done), nrow=10, normalize=True)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, ganconfig.num_epochs, i, len(dataloader_HR), d_loss.item(), g_loss.item())
                )

    dirs = "resources/trained_model/%s/" % (ganconfig.datasetName)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # save generator：
    torch.save(netG.state_dict(), dirs + 'SRGAN.trained_model')
    print('The SRGAN has been trained and saved')