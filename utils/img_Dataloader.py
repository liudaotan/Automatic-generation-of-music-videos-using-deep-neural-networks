import utils.dcgan_config as ganconfig
import utils.srgan_config as srganconfig
import torch
import torchvision.transforms as transforms
from torchvision import datasets

# We can use an image folder dataset the way we have it setup.
# Create the dataset
class dcgan_dataloader:
    def loader(self):
        dataset = datasets.ImageFolder(root=ganconfig.dataroot,
                                       transform=transforms.Compose([
                                             transforms.Resize(ganconfig.image_size),
                                             transforms.CenterCrop(ganconfig.image_size),  # padding
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=ganconfig.batch_size,
                                         shuffle=True, num_workers=ganconfig.workers)
        return dataloader

class srgan_dataloader:
    def loader(self):
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset_HR = datasets.ImageFolder(root = srganconfig.dataroot,
                                      transform=transforms.Compose([
                                          transforms.Resize(srganconfig.image_size_HR),
                                          transforms.CenterCrop(srganconfig.image_size_HR),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))

        dataset_LR = datasets.ImageFolder(root=srganconfig.dataroot,
                                      transform=transforms.Compose([
                                          transforms.Resize(srganconfig.image_size_LR),
                                          transforms.CenterCrop(srganconfig.image_size_LR),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))

        # Create the dataloader
        dataloader_HR = torch.utils.data.DataLoader(dataset_HR, batch_size=srganconfig.batch_size,
                                                    shuffle=False, num_workers=srganconfig.workers)

        dataloader_LR = torch.utils.data.DataLoader(dataset_LR, batch_size = srganconfig.batch_size,
                                                    shuffle=False, num_workers = srganconfig.workers)

        return dataloader_LR,dataloader_HR
