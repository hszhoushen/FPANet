from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import time
import librosa
from PIL import Image, ImageEnhance
import os
import random
import math

import librosa.display
import matplotlib.pyplot as plt

normalizer = np.load('../data/audio_feature_normalizer.npy')
mu    = normalizer[0]
sigma = normalizer[1]

def audio_extract(dataset_name, wav_file):
    # Takes a waveform (length 160,000, sampling rate 16,000)
    # extracts filterbank features (size 201 * 64)
    if dataset_name == 'US8K':
        sr=16000
    else:
        sr=16000

    wav = librosa.load(wav_file, sr=sr)[0]

    if dataset_name == 'US8K':
        # for US8K
        # n_fft = 2048 exp
        # n_fft = 4096 exp
        spec = librosa.core.stft(wav, n_fft=2048,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
    else:
        # for esc
        spec = librosa.core.stft(wav, n_fft=4096,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
    #print('spec.shape:', spec.shape)


    mel = librosa.feature.melspectrogram(S = np.abs(spec), sr = sr, n_mels = 64, fmax = 8000)
    # print('mel.shape:', mel.shape)
    logmel = librosa.core.power_to_db(mel[:, :400])
    # print('logmel.shape:', logmel.shape)
    # mfcc = librosa.feature.mfcc(y=wav, sr=sr, hop_length=400, n_mfcc=20)
    # print('mfcc.shape:', mfcc.shape)

    return logmel.T.astype('float32')

#audio_extract with noise (Gaussian)
def audio_extract_noise(dataset_name, wav_file, SNR):
    if dataset_name == 'US8K':
        sr=16000
    else:
        sr=16000

    wav = librosa.core.load(wav_file, sr=sr)[0]

    '''
    STD_n= 0.001
    noise=np.random.normal(0, STD_n, wav.shape[0])
    wav = wav + noise
    '''
    noise = np.random.randn(len(wav))
    noise = noise-np.mean(noise)
    signal_power = np.linalg.norm(wav - wav.mean())**2 / wav.size
    noise_variance = signal_power/np.power(10,(SNR/10))
    noise = (np.sqrt(noise_variance)/np.std(noise))*noise
    wav = noise + wav

    if dataset_name == 'US8K':
        # for US8K
        # n_fft = 2048 exp
        # n_fft = 4096 exp
        spec = librosa.core.stft(wav, n_fft=2048,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
    else:
        # for esc
        spec = librosa.core.stft(wav, n_fft=4096,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
    #print('spec.shape:', spec.shape)


    mel = librosa.feature.melspectrogram(S = np.abs(spec), sr = sr, n_mels = 64, fmax = 8000)
    # print('mel.shape:', mel.shape)
    logmel = librosa.core.power_to_db(mel[:, :400])
    # print('logmel.shape:', logmel.shape)
    # mfcc = librosa.feature.mfcc(y=wav, sr=sr, hop_length=400, n_mfcc=20)
    # print('mfcc.shape:', mfcc.shape)

    return logmel.T.astype('float32')


def sound_plot(dataset_name, sound_path, label):
    # y (80000, ), sr 16000
    y, sr = librosa.load(sound_path, sr=16000, mono=True, offset=0.0, duration=None)

    spec = librosa.core.stft(y, n_fft=4096,
                             hop_length=400, win_length=1024,
                             window='hann', center=True, pad_mode='constant')
    spec = np.abs(spec)

    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    print('spec.shape:', spec.shape)

    mel = librosa.feature.melspectrogram(S=spec, sr=sr, n_mels=64, fmax=8000)
    print('mel.shape:', mel.shape)

    logmel = librosa.core.power_to_db(mel[:, :400], ref=np.max)
    print('logmel.shape:', logmel.shape)

    # librosa.display.waveplot(y, sr=sr)
    fig = plt.figure()
    fig.add_subplot(221)
    plt.plot(y);
    plt.title('Sound Signal');
    plt.xlabel('Time (samples)');
    plt.ylabel('Amplitude');

    fig.add_subplot(222)
    plt.plot(spec)
    plt.title('Spectrum');
    plt.xlabel('Frequency Bin');
    plt.ylabel('Amplitude');

    fig.add_subplot(223)
    plt.plot(spec_db)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log');
    plt.colorbar(format='%+2.0f dB');
    plt.title('Spectrogram');

    fig.add_subplot(224)
    plt.plot(logmel)
    librosa.display.specshow(logmel, y_axis='mel', fmax=8000, x_axis='time');
    plt.title('Mel Spectrogram');
    plt.colorbar(format='%+2.0f dB');
    plt.show()

def sound_inference(dataset_name, sound_path):

    y, sr = librosa.load(sound_path, sr=16000, mono=True, offset=0.0, duration=None)

    # librosa.display.waveplot(y, sr=sr)
    # plt.show()

    sound = audio_extract(dataset_name, sound_path)
    sound = ((sound - mu) / sigma).astype('float32')
    print('sound.shape:', sound.shape)
    if dataset_name == 'US8K':
        # pad
        if (sound.shape[0] < 161):
            pad_width = 161 - sound.shape[0]
            # print('pad_width:', pad_width)
            sound = np.pad(sound, ((pad_width, 0), (0, 0)), 'constant')
        # del
        elif (sound.shape[0] > 161):
            del_index = range(161, sound.shape[0])
            print('del_index:', del_index)
            sound = np.delete(sound, del_index, axis=0)  # axis=1 删除列，axis=0 删除行

    sound = np.expand_dims(sound, 0)
    sound = np.expand_dims(sound, 0)

    return np.asarray(sound).astype(float)


def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class CVS_Audio_Noise(Dataset):
    def __init__(self, args, data_dir, data_sample, data_type='train', enhance=True, SNR = 0):  # 'train' 'val' 'test'
        self.dataset_name = args.dataset_name
        self.data_dir = data_dir
        self.data_type = data_type
        self.SNR = SNR
        print('self.data_dir:', self.data_dir)
        self.train_len = len(data_sample[0])
        self.test_len = len(data_sample[2])
        self.data_len = self.train_len + self.test_len

        print('data_sample:', self.train_len, self.test_len)

        if data_type == 'train':
            self.data_sample = data_sample[0]
            self.data_label = data_sample[1]

        elif data_type == 'test':
            self.data_sample = data_sample[2]
            self.data_label = data_sample[3]

        self.enable_enhancement = enhance


    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, item):
        # path of visual data
        
        idx = item
        sound_path = os.path.join(self.data_dir, self.data_sample[idx])
        # print('sound_path:', sound_path)

        label = self.data_label[idx]
        # print('label:', label)

        if (self.data_type == 'test'):
            sound = audio_extract_noise(self.dataset_name, sound_path, self.SNR)
        else:
            sound = audio_extract(self.dataset_name, sound_path)
        sound = ((sound - mu) / sigma).astype('float32')
        # print('sound.shape:', sound.shape)
        # TODO: sound normalization and padding tech
        #print('sound.shape:', sound.shape)
        if self.dataset_name == 'US8K':
            sound_len = 221

            # sound signal between 4s and 2s
            if(sound.shape[0] >= int(sound_len/2+1) and sound.shape[0] < sound_len):

                # randomly copy a part of sound from original sound
                pad_width = sound_len - sound.shape[0]
                # print('pad_width:', pad_width)
                # print('sound.shape[0]:', sound.shape[0])
                pad_start = random.randint(1, sound.shape[0] - pad_width)

                sound_mirror = sound[pad_start:pad_start+pad_width]
                # print('sound_mirror.shape:', sound_mirror.shape)
                sound = np.concatenate((sound, sound_mirror), axis=0)
                # print('sound_e.shape:', sound_e.shape)

            # sound signal less than 2s
            if(sound.shape[0] < int(sound_len/2+1)):
                # pad_width = 161-sound.shape[0]
                count = int(sound_len / sound.shape[0])
                #print('count:', count)
                sound_e = sound
                sound_mirror = sound

                # To prelong the sound by copying the sound itself
                for i in range(count):
                    sound_e = np.concatenate((sound_e, sound_mirror), axis=0)
                #print('sound_e first:', sound_e.shape)
                # Randomly truncated at a random point

                # Remove the head
                pad_width = sound_e.shape[0] - sound_len
                #print('pad_width:', pad_width)

                pad_start = random.randint(1, pad_width)
                #print('pad_start:', pad_start)
                del_index = range(0,pad_start)
                #print('del_index:', del_index)
                sound_e = np.delete(sound_e, del_index, axis=0)
                #print('sound_e after 1:', sound_e.shape)

                # Remove the tail
                del_index = range(sound_len, sound_e.shape[0])
                sound = np.delete(sound_e, del_index, axis=0)  # axis=1 删除列，axis=0 删除行
                #print('sound_e after 2:', sound.shape)


            # del
            elif(sound.shape[0] > sound_len):
                del_index = range(sound_len, sound.shape[0])
                #print('del_index:', del_index)
                sound = np.delete(sound, del_index, axis=0)  # axis=1 删除列，axis=0 删除行

        sound = np.expand_dims(sound, 0)
        # print('sound:', sound.shape)

        return np.asarray(sound).astype(float), label   #, self.data_sample[item]

class CVSDataset(Dataset):

    def __init__(self, data_dir, data_sample, data_label, seed, enhance=True, use_KD=True, event_label_name='event_label_bayes'): # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample
        self.data_label = data_label
        self.enable_enhancement = enhance
        self.index_list = [i for i in range(len(self.data_label))]
        self.use_KD = use_KD
        self.event_label_name = event_label_name
        self.seed = seed
        np.random.seed(seed)
        np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
        # return (img_data,audio_data,label)
        #np.random.seed(self.seed)
        #np.random.shuffle(self.index_list)

        image_path = os.path.join(self.data_dir, self.data_sample[self.index_list[item]]+'.jpg')
        print('image_path:', image_path)
        sound_path = os.path.join(self.data_dir, self.data_sample[self.index_list[item]]+'.wav')

        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))

        if self.enable_enhancement:
            image = augment_image(image)

        image = np.asarray(image).astype(float)
        image = np.transpose(image, (2,0,1))
        #print(image.shape)
        sound = audio_extract(sound_path)
        sound = ((sound - mu) / sigma).astype('float32')

        # TODO: sound normalization

        sound = np.expand_dims(sound, 0)
        #sound = ((sound - mu) / sigma).astype('float32')

        scene_label = self.data_label[self.index_list[item]]

        if self.use_KD:
            event_path = os.path.join(self.data_dir, self.event_label_name, self.data_sample[self.index_list[item]]+'_ser.npy')
            event_label = np.load(event_path)

            corr_path = os.path.join('prior_knowledge_pca', 'salient_event_for_%d.npy' % scene_label)
            silent_corr = np.load(corr_path)
            silent_corr = silent_corr[0,:]
            return np.asarray(image).astype(float),np.asarray(sound).astype(float), scene_label, event_label, silent_corr
        else:
            return np.asarray(image).astype(float),np.asarray(sound).astype(float), scene_label


class CVS_Visual(Dataset):
    def __init__(self, data_dir, data_sample, data_type='train', enhance=True):  # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample
        print('data_sample:', len(data_sample), len(data_sample[0]), len(data_sample[1]), len(data_sample[2]), len(data_sample[3]))
        print('data_sample[0]:', data_sample[0])

        if data_type == 'train':
            self.data_sample = data_sample[0]
            self.data_label = data_sample[1]

        elif data_type == 'test':
            self.data_sample = data_sample[2]
            self.data_label = data_sample[3]

        self.enable_enhancement = enhance

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, item):
        # path of visual data
        image_path = os.path.join(self.data_dir, self.data_sample[item] + '.jpg')

        # print('image_path', image_path)
        # print('sound_path:', sound_path)
        if not os.path.isfile(image_path):
            print('image_path:', image_path)
            image_path = os.path.join(self.data_dir, self.data_sample[item] + '.JPG')
            print('image_path:', image_path)
            if not os.path.isfile(image_path):
                image_path = os.path.join(self.data_dir, self.data_sample[item] + '.png')


        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        if self.enable_enhancement:
            image = augment_image(image)

        image = np.asarray(image).astype(float)
        image = np.transpose(image, (2, 0, 1))

        label = self.data_label[item]
        # print('sound:', sound.shape)
        # print('label:', label)

        return np.asarray(image).astype(float), label  # , self.data_sample[item]

