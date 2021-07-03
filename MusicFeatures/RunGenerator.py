import torch
import matplotlib.pyplot as plt
import GAN
import ganconfig
import torchvision.utils as vutils
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model = GAN.Generator(ngpu)
model.load_state_dict(torch.load("../GAN/prettyFace_Generater_size64.pth", map_location=device))
model.to(device)

torch.manual_seed(0)
fixed_noise = torch.randn(64, ganconfig.nz, 1, 1, device=device)
img_list = []
difference = fixed_noise[26,...] - fixed_noise[29, ...]
fixed_noise = difference + fixed_noise
plt.plot(difference.cpu().squeeze())
plt.show()
with torch.no_grad():
    fake = model(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake[:64], nrow=8, padding=2, normalize=True))

fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
latent_features = fixed_noise.squeeze()
smile_set = latent_features[[12, 26, 35, 37, 57],:]
normal_set = latent_features[[4, 5, 21, 63, 39],:]
subtract_list = []
for i in range(5):
    subtract = normal_set[i,:] - smile_set[:, :]
    subtract_list.extend(subtract.tolist())

plot_data = torch.cat((smile_set.cpu(), normal_set.cpu()), dim=0).numpy()
for item in subtract_list:
    plt.plot(item)
plt.show()


pca = PCA(n_components=10)
pca.fit(plot_data)
pca_data = pca.transform(latent_features.cpu().numpy())
sns.heatmap(pca_data)
plt.show()
