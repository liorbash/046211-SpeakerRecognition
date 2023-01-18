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
* A STFT was then applied to the waveform 
* The resulting spectogram was then resized to fit the resnet model expected size (224,224)
* The spectogram was the concatunated on itself to create a (3,224,224) Tenzor
* The Tenzor is fed into the Resnet-18 model


Block diagram:
![image](https://user-images.githubusercontent.com/74953952/213184457-3aea7f2a-2901-48b1-ac79-8e41f27cd8d5.png)


In addition in order to improve the models results we preformed additional preprocessing on the data and changed the resnet-18 model, we will go over the changes and thair effect in the Training and testing different augmetations section of this file. 


### The Dataset

In our project we used the VoxCeleb1 dataset.
VoxCeleb1 is a dataset of speech snippets used for training and evaluating speaker recognition systems. It was created by researchers at the University of Oxford and consists of over 100,000 clips of audio from over 1,251 celebrities. The dataset includes a diverse set of speakers with different accents, ages, and genders, and includes both studio and telephone-quality recordings. The dataset is widely used in the research community for training and evaluating speaker recognition systems, and has been used to train state-of-the-art models. It was the first large-scale public dataset for speaker recognition and has been followed by VoxCeleb2 and VoxCeleb1-E.

Due to computational and memory restrictions, our model was trained on the first 200 speakers (id = 1 to id = 200) but will work on a larger section of the dataset. 

### Training and testing different augmetations





### Prerequisits
| Library  | Version |
| ------------- | ------------- |
| `Python`  | `3.8.16`  |
| `torch`  | `1.13.0`  |
| `numpy`  | `1.21.6`  |
| `torchaudio`  | `0.13.0`  |
| `torchvision`  | `0.14.0`  |
| `pandas`  | `1.3.5`  |
| `librosa`  | `0.8.1`  |
| `matplotlib`  | `3.2.2`  |
| `logzero` | `1.7.0`  |

### Running our model

### References
We based our project on the results of the following papers and github repositories:
*S. Bianco, E. Cereda and P. Napoletano, "Discriminative Deep Audio Feature Embedding for Speaker Recognition in the Wild," 2018 IEEE 8th International Conference on Consumer Electronics - Berlin (ICCE-Berlin), Berlin, Germany, 2018, pp. 1-5, doi: 10.1109/ICCE-Berlin.2018.8576237.
*M. Jakubec, E. Lieskovska and R. Jarina, "Speaker Recognition with ResNet and VGG Networks," 2021 31st International Conference Radioelektronika (RADIOELEKTRONIKA), Brno, Czech Republic, 2021, pp. 1-5, doi: 10.1109/RADIOELEKTRONIKA52220.2021.9420202.
*https://github.com/samtwl/Deep-Learning-Contrastive-Center-Loss-Transfer-Learning-Food-Classification-/tree/master
