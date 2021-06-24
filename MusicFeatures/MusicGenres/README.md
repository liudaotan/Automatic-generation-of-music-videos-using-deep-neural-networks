# Music genre classification

This module classifies the music genres which can be used as the feature of music. The dataset used in this module is [FMA dataset](https://github.com/mdeff/fma) which has more than 100k piece of audio. This project used FMA_small that contains 8,000 piece of audio and train the model. We split the data set into a training set and validation set, whose training set accounts for 85% of the data set.

The audio files should be decompressed to the '/fma_small', and the .csv files should be placed in the '/fma_metadata'. The preprocess.py preprocesses the data and saves it to the '/processed_data'. Hence, preprocess.py should be executed first. 

### Model structure:
This module used 2 models (CNN, CRNN) to train the classifier but now the project is using the CRNN ([Shi et al. 2016](https://arxiv.org/pdf/1507.05717.pdf)) as the classifier because it performs much better than CNN. The model used the CRNN architecture proposed by [Choi K, Fazekas G, Sandler M, et al](https://arxiv.org/pdf/1609.04243.pdf) in 2016. More models will be applied in the future, such as fusion spectrogram model ([Weiping et al.](http://dcase.community/documents/workshop2017/proceedings/DCASE2017Workshop_Zheng_159.pdf)).


### Performance

the performance of the validation set.

| epoch\model |  cnn  |  crnn | c2rnn |
|:-----------:|:-----:|:-----:|:-----:|
|      10     | 0.349 | 0.521 | 0.527 |
|      20     | 0.447 | 0.573 | 0.619 |
|      50     | 0.482 | 0.561 | 0.629 |

### Predict
run predict.py
```commandline
python predict.py -f <filepath>
```

### Citation

[1]. Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE transactions on pattern analysis and machine intelligence, 39(11), 2298-2304.

[2]. Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017, March). Convolutional recurrent neural networks for music classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 2392-2396). IEEE.

[3]. Weiping, Z., Jiantao, Y., Xiaotao, X., Xiangtao, L., & Shaohu, P. (2017). Acoustic scene classification using deep convolutional neural network and multiple spectrograms fusion. Detection and Classification of Acoustic Scenes and Events (DCASE).
