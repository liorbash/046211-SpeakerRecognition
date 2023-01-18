# 046211-SpeakerRecognition

## Agenda
* Project overview
* The Dataset
* Training and testing different augmetations
* Prerequisits
* Running our model
* Credits and References


### Project overview

Speaker recognition is the process of identifying or verifying the identity of a speaker based on their voice. It is commonly used for security and authentication purposes, such as access control to secure buildings or computer systems. The main challenges in speaker recognition include variability in the speech signal due to factors such as speaking style, background noise, and microphone quality, as well as the need for large amounts of training data to accurately model an individual's speech patterns. Additionally, speaker recognition systems must be able to adapt to changes in a person's voice over time, such as due to aging or changes in health.

In our project, we implemented a resnet-18 based nural network that trained on the VoxCeleb1 dataset. In order to adapt the audio files to fit the network we preformed the following augmentations:

* Every audiofile was clipped or padded into the same length 
* A STFT was then applied to the file 
* The resulting spectogram was then resized to fit the resnet model expected size (224,224)
* The spectogram was the concatunated on itself to create a (3,224,224) Tenzor
* The Tenzor is fed into the Resnet-18 model


ADD Block diagram


In addition in order to improve the models results we preformed additional preprocessing on the data and changed the resnet-18 model, we will go over the changes and thair effect in the Training and testing different augmetations section of this file. 


### The Dataset

In our project we used the VoxCeleb1 dataset.
VoxCeleb1 is a dataset of speech snippets used for training and evaluating speaker recognition systems. It was created by researchers at the University of Oxford and consists of over 100,000 clips of audio from over 1,251 celebrities. The dataset includes a diverse set of speakers with different accents, ages, and genders, and includes both studio and telephone-quality recordings. The dataset is widely used in the research community for training and evaluating speaker recognition systems, and has been used to train state-of-the-art models. It was the first large-scale public dataset for speaker recognition and has been followed by VoxCeleb2 and VoxCeleb1-E.


### Training and testing different augmetations





### Prerequisits -- add versions and any other modules we use
```
python -- version 
numpy
sklearn
torch
torchaudio
torchvision
librosa
matplotlib
time
copy
pandas
random
```
