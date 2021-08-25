# Automatic-generation-of-music-videos-using-deep-neural-networks


A group in the department is actively working on artificial intelligence (AI) technologies for the generation of music using deep neural networks. To complement this, we would like to extend our work to generate a visual accompaniment to the music. Recent progress in deep neural networks allows generation of images using architectures such as the generative adversarial network (GAN), a particular approach in which two neural networks contest with each other. Given a set of training images, the GAN learns to generate new images with the same statistics as the training set.

You can get some idea of what is required from the following commercial system: https://wzrd.ai/. We think that improvements can be made to make the video more tightly synchronised to the music.

Key stages in this project will be to implement a GAN using open source toolkits and inform image generation using an acoustic analysis of the music signal (e.g. by extracting features related to timbre, pitch, amplitude etc). It is anticipated that the system will be made available via a web service and will be evaluated by a small user group (and will therefore require ethics approval for user testing).

Key requirements

Upload audio as wav file (no need to support other formats)
Select a number of keyframes generated by the GAN
Provide options for what aspects of the sound file influence the video generation (sound level, frequency content, timbre)
Investigate various mechanisms for making the video generation depend on sound quality (pre-process the input to the network, post-process the output from the network, change network parameters)
We need a suitable image library for training the network and for use in the system; fairly abstract images (not faces) perhaps of landscape photographs, city scapes etc. For example: type “abstract art” into google image search, use a browser extension to download all the thumbnail images. 
Maybe a number of different image sets for generation based on different themes (the network could be trained on more general image data)
Ideally implement this within a web service so that users can upload a sound file and download the video
Do video generation from the GAN output

Suggested split of work (open to discussion)

Qibin (MSc CS+SLP)
Audio feature extraction

Daotan (MSc DA)
GAN implementation

---

### Usage
#### requirements:

    pytorch>1.7.0
    torchaudio
    torchvision
    librosa
    ffmpeg-python

Before executing the program, please put the audio file to the 'resources/music' folder.

#### Command:

    python main.py --help

    outputs:

    Auto music video generator
    
    optional arguments:
      -h, --help            show this help message and exit
      --audio AUDIO         audio file path
      --model MODEL         the model used to generate the video (option: 'landscape', 'abstract', 'pretty_face', 'face512')
      --method METHOD       the method applied to the change of video (option: 'base' or 'hpss')
      --emphasize EMPHASIZE

The program uses the base generator and landscape GAN by default. If you don't want to make any customization, you can simply run:
    
    python main.py --audio 'audio'

'audio' is the name of the audio file. 
