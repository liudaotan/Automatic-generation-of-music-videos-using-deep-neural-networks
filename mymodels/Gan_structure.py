import torch as tr
import torch.nn as nn
import utils.dcgan_config as ganconfig
import torch.nn.functional as f

class DCGAN_Generator(tr.nn.Module):
    def __init__(self, ngpu):
        super(DCGAN_Generator, self).__init__()
        self.ngpu = ngpu
        self.main = tr.nn.Sequential(
            # input is Z, going into a convolution
            tr.nn.ConvTranspose2d(ganconfig.nz, ganconfig.ngf * 8, 4, 1, 0, bias=False),
            tr.nn.BatchNorm2d(ganconfig.ngf * 8),
            tr.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            tr.nn.ConvTranspose2d(ganconfig.ngf * 8, ganconfig.ngf * 4, 4, 2, 1, bias=False),
            tr.nn.BatchNorm2d(ganconfig.ngf * 4),
            tr.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            tr.nn.ConvTranspose2d(ganconfig.ngf * 4, ganconfig.ngf * 2, 4, 2, 1, bias=False),
            tr.nn.BatchNorm2d(ganconfig.ngf * 2),
            tr.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            tr.nn.ConvTranspose2d(ganconfig.ngf * 2, ganconfig.ngf, 4, 2, 1, bias=False),
            tr.nn.BatchNorm2d(ganconfig.ngf),
            tr.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            tr.nn.ConvTranspose2d(ganconfig.ngf, ganconfig.nc, 4, 2, 1, bias=False),
            tr.nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class DCGAN_Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(DCGAN_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(ganconfig.nc, ganconfig.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ganconfig.ndf, ganconfig.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ganconfig.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ganconfig.ndf * 2, ganconfig.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ganconfig.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ganconfig.ndf * 4, ganconfig.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ganconfig.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ganconfig.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SRGAN_Generator(tr.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tr.nn.Conv2d(3, 64, 9, padding=4, bias=False)
        self.conv2 = tr.nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv3_1 = tr.nn.Conv2d(64, 256, 3, padding=1, bias=False)
        self.conv3_2 = tr.nn.Conv2d(64, 256, 3, padding=1, bias=False)
        self.conv4 = tr.nn.Conv2d(64, 3, 9, padding=4, bias=False)
        self.bn = tr.nn.BatchNorm2d(64)
        self.ps = tr.nn.PixelShuffle(2)
        self.prelu = tr.nn.PReLU()

    def forward(self, x):
        block1 = self.prelu(self.conv1(x))
        block2 = tr.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block1))))), block1)
        block3 = tr.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block2))))), block2)
        block4 = tr.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block3))))), block3)
        block5 = tr.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block4))))), block4)
        block6 = tr.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block5))))), block5)
        block7 = tr.add(self.bn(self.conv2(block6)), block1)
        block8 = self.prelu(self.ps(self.conv3_1(block7)))
        block9 = self.prelu(self.ps(self.conv3_2(block8)))
        block10 = self.conv4(block9)
        return block10


class SRGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.drop = nn.Dropout2d(0.3)

    def forward(self, x):
        block1 = f.leaky_relu(self.conv1(x))
        block2 = f.leaky_relu(self.bn2(self.conv2(block1)))
        block3 = f.leaky_relu(self.bn3(self.conv3(block2)))
        block4 = f.leaky_relu(self.bn4(self.conv4(block3)))
        block5 = f.leaky_relu(self.bn5(self.conv5(block4)))
        block6 = f.leaky_relu(self.bn6(self.conv6(block5)))
        block7 = f.leaky_relu(self.bn7(self.conv7(block6)))
        block8 = f.leaky_relu(self.bn8(self.conv8(block7)))
        block8 = block8.view(-1, block8.size(1) * block8.size(2) * block8.size(3))
        block9 = f.leaky_relu(self.fc1(block8), )
        #       block9 = block9.view(-1,block9.size(1)*block9.size(2)*block9.size(3))
        block10 = tr.sigmoid(self.drop(self.fc2(block9)))
        return block10

class GAN_Generators(tr.nn.Module):

    def __init__(self, base_pth, boost_pth, ngpu=1):
        """
        Parameters
        ----------
        base_pth: the path of the pretrained DCGAN
        boost_pth:the path of the pretrained SRGAN

        """
        super(GAN_Generators, self).__init__()
        self.dc_generator = DCGAN_Generator(ngpu=ngpu)
        self.sr_generator = SRGAN_Generator()

        self.dc_generator.load_state_dict(tr.load(base_pth))
        self.sr_generator.load_state_dict(tr.load(boost_pth))

    def forward(self, x):
        # concat the two GANs together
        x = self.dc_generator(x)
        x = self.sr_generator(x)
        return x