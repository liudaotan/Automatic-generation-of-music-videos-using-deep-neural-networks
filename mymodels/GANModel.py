import torch as tr
import utils.ganconfig as ganconfig

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