# Music genre classification

This module classifies the music genres which can be used as the feature of music. The dataset used in this module is [FMA dataset](https://github.com/mdeff/fma) which has more than 100k piece of audio. This project used FMA_small that contains 8,000 piece of audio and train the model. 

The audio files should be decompressed to the '/fma_small', and the .csv files should be place in the '/fma_metadata'. The preprocess.py preprocesses the data and save it to the '/processed_data'. Hence, preprocess.py should be executed first.

### Model structure:
This module used 2 model(CNN, CRNN) to train the classifier but now the project is using the [CRNN (Shi et al. 2016)](https://arxiv.org/pdf/1507.05717.pdf) as the classifier because it performs much better than CNN. The model used the CRNN architecture proposed by [Choi K, Fazekas G, Sandler M, et al](https://arxiv.org/pdf/1609.04243.pdf) in 2016

### Citation

[1]. Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE transactions on pattern analysis and machine intelligence, 39(11), 2298-2304.

[2]. Choi K, Fazekas G, Sandler M, et al. Convolutional recurrent neural networks for music classification[C]//2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017: 2392-2396.
