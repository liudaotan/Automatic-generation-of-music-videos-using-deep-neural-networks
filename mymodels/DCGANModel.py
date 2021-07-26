import torch as tr
import utils.ganconfig as ganconfig

class Generator(tr.nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
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
