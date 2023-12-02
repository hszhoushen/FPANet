from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import time
import librosa
from PIL import Image, ImageEnhance
import os
import random

import librosa.display
import matplotlib.pyplot as plt

normalizer = np.load('./data/audio_feature_normalizer.npy')
print('len(normalizer:', len(normalizer))
mu    = normalizer[0]
print('mu:', mu)
sigma = normalizer[1]
print('sigma:', sigma)




def audio_extract(dataset_name, wav_file):
    # Takes a waveform (length 160,000, sampling rate 16,000)
    # extracts filterbank features (size 201 * 64)
    if dataset_name == 'US8K':
        sr = 16000
    elif dataset_name == 'DCASE2019' or dataset_name == 'DCASE2021-Audio':
        sr = 48000
    elif dataset_name == 'HomeAutomation':
        sr = 16000
    else:
        sr = 16000

    wav = librosa.load(wav_file, sr=sr)[0]

    if dataset_name == 'US8K':
        # for US8K
        # n_fft = 2048 exp
        # n_fft = 4096 exp
        spec = librosa.core.stft(wav, n_fft=2048,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
        mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=sr, n_mels=64, fmax=8000)
        logmel = librosa.core.power_to_db(mel[:, :400])

    elif dataset_name == 'HomeAutomation':
        # for esc
        spec = librosa.core.stft(wav, n_fft=2048,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
        mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=sr, n_mels=64, fmax=8000)
        # print('mei.shape:', mel.shape)
        logmel = librosa.core.power_to_db(mel[:, :400])
        # print('logmel.shape:', logmel.shape)

    elif dataset_name == 'DCASE2019' or dataset_name == 'DCASE2021-Audio':
        # for DCASE2019
        spec = librosa.core.stft(wav, n_fft=2048,
                                 hop_length=512, win_length=512,
                                 window='hann', center=True, pad_mode='constant')
        # print('spec.shape:', spec.shape)

        mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=sr, n_mels=64, fmax=8000)
        logmel = librosa.core.power_to_db(mel[:, :862])

    else:
        # for esc
        spec = librosa.core.stft(wav, n_fft=4096,
                                 hop_length=400, win_length=1024,
                                 window='hann', center=True, pad_mode='constant')
        mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=sr, n_mels=64, fmax=8000)
        logmel = librosa.core.power_to_db(mel[:, :300])


    # print('logmel.shape:', logmel.shape)
    # mfcc = librosa.feature.mfcc(y=wav, sr=sr, hop_length=400, n_mfcc=20)
    # print('mfcc.shape:', mfcc.shape)

    return logmel.T.astype('float32')

def noise_produce(wav, SNR):
    # Set a target SNR
    target_snr_db = SNR

    # # noise = np.random.randn(len(wav))
    # noise = np.random.normal(0, 1, len(wav))
    # noise = noise - np.mean(noise)                                      # normalize

    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(wav)
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(wav))

    # Noise up the original signal
    y_volts = wav + noise_volts

    # signal_power = np.linalg.norm(wav - wav.mean())**2 / wav.size       #
    # noise_variance = signal_power / np.power(10, (SNR/10))
    # noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return y_volts

#audio_extract with noise (Gaussian)
def audio_extract_noise(dataset_name, wav_file, SNR):
    if dataset_name == 'US8K':
        sr=16000
    else:
        sr=16000

    wav = librosa.load(wav_file, sr=sr)[0]

    # STD_n= 0.001
    # noise=np.random.normal(0, STD_n, wav.shape[0])
    # wav = wav + noise

    wav = noise_produce(wav, SNR)

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

    return np.asarray(sound).astype('float32')


def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image





class CVS_Audio_Noise(Dataset):
    def __init__(self, args, data_dir, data_sample, data_type='train', enhance=True):  # 'train' 'val' 'test'
        self.dataset_name = args.dataset_name
        self.data_dir = data_dir
        print('self.data_dir:', self.data_dir)
        self.train_len = len(data_sample[0])
        self.test_len = len(data_sample[2])
        self.data_len = self.train_len + self.test_len
        self.SNR = 20

        print('data_sample:', self.train_len, self.test_len)
        self.data_type = data_type
        if data_type == 'train':
            self.data_sample = data_sample[0]
            self.data_label = data_sample[1]

        elif data_type == 'test':
            self.data_sample = data_sample[2]
            self.data_label = data_sample[3]

        self.enable_enhancement = enhance

    def esc10_label_rename(self, label):
        if self.dataset_name == 'ESC10':
            if label == 41:
                label = 9
            elif label == 12:
                label = 4
            elif label == 40:
                label = 8
            elif label == 10:
                label = 2
            elif label == 20:
                label = 5
            elif label == 38:
                label = 7
            elif label == 21:
                label = 6
            elif label == 11:
                label = 3
        return label

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, item):
        # path of visual data

        idx = item
        sound_path = os.path.join(self.data_dir, self.data_sample[idx])
        # print('sound_path:', sound_path)

        label = self.data_label[idx]
        label = self.esc10_label_rename(label)
        if (self.data_type == 'test'):
            sound = audio_extract_noise(self.dataset_name, sound_path, self.SNR)
        else:
            sound = audio_extract(self.dataset_name, sound_path)
        # sound = audio_extract(self.dataset_name, sound_path)
        sound = ((sound - mu) / sigma).astype('float32')
        # print('sound.shape:', sound.shape)
        # TODO: sound normalization and padding tech
        # print('sound.shape:', sound.shape)
        if self.dataset_name == 'US8K':
            sound_len = 221
            # sound signal between 4s and 2s
            if (sound.shape[0] >= int(sound_len / 2 + 1) and sound.shape[0] < sound_len):
                # randomly copy a part of sound from original sound
                pad_width = sound_len - sound.shape[0]
                # print('pad_width:', pad_width)
                # print('sound.shape[0]:', sound.shape[0])
                pad_start = random.randint(1, sound.shape[0] - pad_width)

                sound_mirror = sound[pad_start:pad_start + pad_width]
                # print('sound_mirror.shape:', sound_mirror.shape)
                sound = np.concatenate((sound, sound_mirror), axis=0)
                # print('sound_e.shape:', sound_e.shape)

            # sound signal less than 2s
            if (sound.shape[0] < int(sound_len / 2 + 1)):
                # pad_width = 161-sound.shape[0]
                count = int(sound_len / sound.shape[0])
                # print('count:', count)
                sound_e = sound
                sound_mirror = sound

                # To prelong the sound by copying the sound itself
                for i in range(count):
                    sound_e = np.concatenate((sound_e, sound_mirror), axis=0)
                # print('sound_e first:', sound_e.shape)
                # Randomly truncated at a random point

                # Remove the head
                pad_width = sound_e.shape[0] - sound_len
                # print('pad_width:', pad_width)

                pad_start = random.randint(1, pad_width)
                # print('pad_start:', pad_start)
                del_index = range(0, pad_start)
                # print('del_index:', del_index)
                sound_e = np.delete(sound_e, del_index, axis=0)
                # print('sound_e after 1:', sound_e.shape)

                # Remove the tail
                del_index = range(sound_len, sound_e.shape[0])
                sound = np.delete(sound_e, del_index, axis=0)  # axis=1 删除列，axis=0 删除行
                # print('sound_e after 2:', sound.shape)


            # del
            elif (sound.shape[0] > sound_len):
                del_index = range(sound_len, sound.shape[0])
                # print('del_index:', del_index)
                sound = np.delete(sound, del_index, axis=0)  # axis=1 删除列，axis=0 删除行

        sound = np.expand_dims(sound, 0)
        # print('sound:', sound.shape)

        return np.asarray(sound).astype('float32'), label, idx  # , self.data_sample[item]

class CVS_Audio(Dataset):
    def __init__(self, args, data_dir, data_sample, data_type='train', enhance=True):  # 'train' 'val' 'test'
        self.dataset_name = args.dataset_name
        self.data_dir = data_dir
        print('self.data_dir:', self.data_dir)
        self.train_len = len(data_sample[0])
        self.test_len = len(data_sample[2])
        self.data_len = self.train_len + self.test_len

        print('train:', self.train_len, 'test:', self.test_len)

        if data_type == 'train':
            self.data_sample = data_sample[0]
            self.data_label = data_sample[1]

        elif data_type == 'test':
            self.data_sample = data_sample[2]
            self.data_label = data_sample[3]

        self.enable_enhancement = enhance

    def esc10_label_rename(self, label):
        if label == 41:
            label = 9
        elif label == 12:
            label = 4
        elif label == 40:
            label = 8
        elif label == 10:
            label = 2
        elif label == 20:
            label = 5
        elif label == 38:
            label = 7
        elif label == 21:
            label = 6
        elif label == 11:
            label = 3
        return label

    def dcase2019_label_name(self, label):
        if (label == 'airport'):
            label = 0
        elif (label == 'bus'):
            label = 1
        elif (label == 'metro'):
            label = 2
        elif (label == 'metro_station'):
            label = 3
        elif (label == 'park'):
            label = 4
        elif (label == 'public_square'):
            label = 5
        elif (label == 'shopping_mall'):
            label = 6
        elif (label == 'street_pedestrian'):
            label = 7
        elif (label == 'street_traffic'):
            label = 8
        elif (label == 'tram'):
            label = 9
        return label

    def dcase2021_label_name(self, label):
        if (label == 'airport'):
            label = 0
        elif (label == 'bus'):
            label = 1
        elif (label == 'metro'):
            label = 2
        elif (label == 'metro_station'):
            label = 3
        elif (label == 'park'):
            label = 4
        elif (label == 'public_square'):
            label = 5
        elif (label == 'shopping_mall'):
            label = 6
        elif (label == 'street_pedestrian'):
            label = 7
        elif (label == 'street_traffic'):
            label = 8
        elif (label == 'tram'):
            label = 9
        return label

    def __len__(self):
        return len(self.data_sample)

    def sound_padding(self, sound, sound_len):

        # sound signal between 4s and 2s
        if (sound.shape[0] >= int(sound_len / 2 + 1) and sound.shape[0] < sound_len):
            # randomly copy a part of sound from original sound
            pad_width = sound_len - sound.shape[0]
            # print('pad_width:', pad_width)
            # print('sound.shape[0]:', sound.shape[0])
            pad_start = random.randint(1, sound.shape[0] - pad_width)

            sound_mirror = sound[pad_start:pad_start + pad_width]
            # print('sound_mirror.shape:', sound_mirror.shape)
            sound = np.concatenate((sound, sound_mirror), axis=0)
            # print('sound_e.shape:', sound_e.shape)

        # sound signal less than 2s
        if (sound.shape[0] < int(sound_len / 2 + 1)):
            # pad_width = 161-sound.shape[0]
            count = int(sound_len / sound.shape[0])
            # print('count:', count)
            sound_e = sound
            sound_mirror = sound

            # To prelong the sound by copying the sound itself
            for i in range(count):
                sound_e = np.concatenate((sound_e, sound_mirror), axis=0)
            # print('sound_e first:', sound_e.shape)
            # Randomly truncated at a random point

            # Remove the head
            pad_width = sound_e.shape[0] - sound_len
            # print('pad_width:', pad_width)

            pad_start = random.randint(1, pad_width)
            # print('pad_start:', pad_start)
            del_index = range(0, pad_start)
            # print('del_index:', del_index)
            sound_e = np.delete(sound_e, del_index, axis=0)
            # print('sound_e after 1:', sound_e.shape)

            # Remove the tail
            del_index = range(sound_len, sound_e.shape[0])
            sound = np.delete(sound_e, del_index, axis=0)  # axis=1 删除列，axis=0 删除行
            # print('sound_e after 2:', sound.shape)

        # del
        elif (sound.shape[0] > sound_len):
            del_index = range(sound_len, sound.shape[0])
            # print('del_index:', del_index)
            sound = np.delete(sound, del_index, axis=0)  # axis=1 删除列，axis=0 删除行

        return sound

    def __getitem__(self, item):
        # path of visual data
        idx = item
        sound_path = os.path.join(self.data_dir, self.data_sample[idx])
        # print('sound_path:', sound_path)

        label = self.data_label[idx]
        # print('label:', label)
        if self.dataset_name == 'ESC10':
            label = self.esc10_label_rename(label)
        elif self.dataset_name == 'DCASE2019':
            label = self.dcase2019_label_name(label)
        elif self.dataset_name == 'DCASE2021-Audio':
            label = self.dcase2021_label_name(label)

        # print('label:', label)

        # print('mu:', mu.shape, 'sigma:', sigma.shape)
        if self.dataset_name == 'HomeAutomation':
            sound = audio_extract(self.dataset_name, sound_path)

        elif self.dataset_name == 'DCASE2021-Audio' or self.dataset_name == 'DCASE2019':
            sound = audio_extract(self.dataset_name, sound_path)

        else:
            sound = audio_extract(self.dataset_name, sound_path)
            sound = ((sound - mu) / sigma).astype('float32')

        # print('sound.shape:', sound.shape)
        # TODO: sound normalization and padding tech

        if self.dataset_name == 'US8K':
            sound_len = 221
            sound = self.sound_padding(sound, sound_len)
        elif self.dataset_name == 'HomeAutomation':
            sound_len = 400
            if (sound.shape[0] < 400):
                #print('sound.shape[0]:', sound.shape[0])
                sound = self.sound_padding(sound, sound_len)
                #print('padding sound.shape[0]:', sound.shape[0])

        sound = np.expand_dims(sound, 0)
        # print('sound:', sound.shape)

        return np.asarray(sound).astype('float32'), label, idx   #, self.data_sample[item]


class Visual_Dataset(Dataset):

    def __init__(self, data_dir, data_sample, data_label, enhance=True, use_KD=True, event_label_name='event_label_bayes'): # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.sound_dir = os.path.join(self.data_dir[:-7], 'sound')
        self.data_sample = data_sample
        self.data_label = data_label

        self.enable_enhancement = enhance
        self.index_list = [i for i in range(len(self.data_label))]
        self.use_KD = use_KD
        self.event_label_name = event_label_name
        # self.seed = seed
        # np.random.seed(seed)
        np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
 
        image_path = os.path.join(self.data_dir, self.data_sample[self.index_list[item]][0] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        if self.enable_enhancement:
            image = augment_image(image)
        image = np.asarray(image).astype(float)
        image = np.transpose(image, (2,0,1))

        scene_label = self.data_label[self.index_list[item]]
   
        return np.asarray(image).astype(float), scene_label, image_path


class Audio_Dataset(Dataset):

    def __init__(self, data_dir, data_sample, data_label, enhance=True, use_KD=True, event_label_name='event_label_bayes'): # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.sound_dir = os.path.join(self.data_dir[:-7], 'sound')
        self.data_sample = data_sample
        self.data_label = data_label

        self.enable_enhancement = enhance
        self.index_list = [i for i in range(len(self.data_label))]
        self.use_KD = use_KD
        self.event_label_name = event_label_name

        np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
        sound_path = os.path.join(self.sound_dir, self.data_sample[self.index_list[item]][0] + '.wav')

        sound = audio_extract('ADVANCE', sound_path)
        sound = ((sound - mu) / sigma).astype('float32')

        sound = np.expand_dims(sound, 0)
        #sound = ((sound - mu) / sigma).astype('float32')

        scene_label = self.data_label[self.index_list[item]]


        return np.asarray(sound).astype(float), scene_label, sound_path


class CVSDataset(Dataset):

    def __init__(self, data_dir, data_sample, data_label, enhance=True, use_KD=True, event_label_name='event_label_bayes'): # 'train' 'val' 'test'
        self.data_dir = data_dir
        
        print('(self.data_dir[:]):', (self.data_dir[:]))
        print('(self.data_dir[:-7]):', (self.data_dir[:-7]))
        self.sound_dir = os.path.join(self.data_dir[:-7], 'sound')
        print('(self.sound_dir):', (self.sound_dir))

        self.data_sample = data_sample
        self.data_label = data_label
        # print('self.data_sample:', self.data_sample)
        # print('self.data_label:', self.data_label)

        self.enable_enhancement = enhance
        self.index_list = [i for i in range(len(self.data_label))]
        self.use_KD = use_KD
        self.event_label_name = event_label_name
        # self.seed = seed
        # np.random.seed(seed)
        np.random.shuffle(self.index_list)

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, item):
        # return (img_data,audio_data,label)
        #np.random.seed(self.seed)
        #np.random.shuffle(self.index_list)


        # print('self.data_label:', self.data_label[item])

        # print('item:', item)
        # print('self.data_sample[item]:', self.data_sample[item][0])
        #
        # print('self.index_list:', self.index_list)
        # print('self.data_label:', self.data_label)

        image_path = os.path.join(self.data_dir, self.data_sample[self.index_list[item]][0] + '.jpg')
        
        #image_path = os.path.join(self.data_dir, self.data_sample[self.index_list[item]][0] + '.jpg')

        # print('image_path:', image_path)
        # print('test:', self.data_sample[self.index_list[item]][0].split('.')[0])
        sound_name = self.data_sample[self.index_list[item]][0].split('.')[0] + '.wav'
        sound_path = os.path.join(self.sound_dir, sound_name)
        # print('sound_path:', sound_path)


        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))

        if self.enable_enhancement:
            image = augment_image(image)

        image = np.asarray(image).astype(float)
        image = np.transpose(image, (2,0,1))
        #print(image.shape)
        sound = audio_extract('ADVANCE', sound_path)
        sound = ((sound - mu) / sigma).astype('float32')

        # TODO: sound normalization

        sound = np.expand_dims(sound, 0)
        #sound = ((sound - mu) / sigma).astype('float32')

        scene_label = self.data_label[self.index_list[item]]
        # print('scene_label:', scene_label)
        # if self.use_KD:
        #     event_path = os.path.join(self.data_dir, self.event_label_name, self.data_sample[self.index_list[item]]+'_ser.npy')
        #     event_label = np.load(event_path)
        #
        #     corr_path = os.path.join('prior_knowledge_pca', 'salient_event_for_%d.npy' % scene_label)
        #     silent_corr = np.load(corr_path)
        #     silent_corr = silent_corr[0,:]
        #     return np.asarray(image).astype(float),np.asarray(sound).astype(float), scene_label, event_label, silent_corr
        # else:

        return np.asarray(sound).astype(float),np.asarray(image).astype(float), scene_label, image_path


class CVS_Visual(Dataset):
    def __init__(self, args, data_dir, data_sample, data_type='train', enhance=True):  # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample

        if data_type == 'train':
            self.data_sample = data_sample[4]
            self.data_label = data_sample[1]

        elif data_type == 'test':
            self.data_sample = data_sample[5]
            self.data_label = data_sample[3]

        self.enable_enhancement = enhance

    def dcase2021_label_name(self, label):
        if (label == 'airport'):
            label = 0
        elif (label == 'bus'):
            label = 1
        elif (label == 'metro'):
            label = 2
        elif (label == 'metro_station'):
            label = 3
        elif (label == 'park'):
            label = 4
        elif (label == 'public_square'):
            label = 5
        elif (label == 'shopping_mall'):
            label = 6
        elif (label == 'street_pedestrian'):
            label = 7
        elif (label == 'street_traffic'):
            label = 8
        elif (label == 'tram'):
            label = 9
        return label

    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, item):
        # path of visual data
        image_path = self.data_sample[item]

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
        label = self.dcase2021_label_name(label)

        return np.asarray(image).astype(float), label, image_path  # , self.data_sample[item]

class CVS_Audio_Visual(Dataset):
    def __init__(self, args, data_dir, data_sample, data_type='train', enhance=True):  # 'train' 'val' 'test'
        self.data_dir = data_dir
        self.data_sample = data_sample
        self.dataset_name = args.dataset_name

        if data_type == 'train':
            self.audio_data_sample = data_sample[0]
            self.data_label = data_sample[1]
            self.visual_data_sample = data_sample[4]

        elif data_type == 'test':
            self.audio_data_sample = data_sample[2]
            self.data_label = data_sample[3]
            self.visual_data_sample = data_sample[5]

        self.enable_enhancement = enhance

    def dcase2021_label_name(self, label):
        if (label == 'airport'):
            label = 0
        elif (label == 'bus'):
            label = 1
        elif (label == 'metro'):
            label = 2
        elif (label == 'metro_station'):
            label = 3
        elif (label == 'park'):
            label = 4
        elif (label == 'public_square'):
            label = 5
        elif (label == 'shopping_mall'):
            label = 6
        elif (label == 'street_pedestrian'):
            label = 7
        elif (label == 'street_traffic'):
            label = 8
        elif (label == 'tram'):
            label = 9
        return label

    def __len__(self):
        return len(self.audio_data_sample)

    def __getitem__(self, item):
        # path of visual data
        audio_path = os.path.join(self.data_dir, self.audio_data_sample[item])
        # print('audio_path:', audio_path)
        image_path = self.visual_data_sample[item]
        # print('image_path:', image_path)

        audio = audio_extract(self.dataset_name, audio_path)
        audio = np.expand_dims(audio, 0)

        # if not os.path.isfile(image_path):
        #     print('image_path:', image_path)
        #     image_path = os.path.join(self.data_dir, self.data_sample[item] + '.JPG')
        #     print('image_path:', image_path)
        #     if not os.path.isfile(image_path):
        #         image_path = os.path.join(self.data_dir, self.data_sample[item] + '.png')

        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        if self.enable_enhancement:
            image = augment_image(image)

        image = np.asarray(image).astype(float)
        image = np.transpose(image, (2, 0, 1))

        label = self.data_label[item]
        label = self.dcase2021_label_name(label)

        return np.asarray(audio).astype('float32'), np.asarray(image).astype(float), label, image_path  # , self.data_sample[item]