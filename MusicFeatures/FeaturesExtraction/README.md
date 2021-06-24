# Music features extraction

---
This module obtains the music timbre features such as spectral centroid and power spectal density. This module will also use the [nnAudio](https://github.com/KinWaiCheuk/nnAudio) library to extract the features because it provides features extractions that are implemented by Pytorch. So, we can use the GPU to accelerate the training.

---
### Citation
[1]. K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, “nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks,” in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.
