import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
# Press the green button in the gutter to run the script.
from mymodels.Gan_structure import DCGAN_Generator,DCGAN_Discriminator,weights_init
import utils.dcgan_config as ganconfig
from utils.img_Dataloader import dcgan_dataloader
from utils.seeds import start_seeds

if __name__ == '__main__':

    dataloader = dcgan_dataloader()
    dataloader = dataloader.loader()

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ganconfig.ngpu > 0) else "cpu")

    # Create the generator
    netG = DCGAN_Generator(ganconfig.ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the Discriminator
    netD = DCGAN_Discriminator(ganconfig.ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # the start seeds
    fixed_noise = start_seeds(device).fixed_noise

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=ganconfig.lr, betas=(ganconfig.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=ganconfig.lr, betas=(ganconfig.beta1, 0.999))

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(ganconfig.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, ganconfig.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, ganconfig.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == ganconfig.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


    dirs = "resources/trained_model/%s/" % (ganconfig.datasetName)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # save generator：
    torch.save(netG.state_dict(),dirs + 'DCGAN.trained_model')
    print('The DCGAN has been trained and saved')
