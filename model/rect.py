import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages')
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.patches import Circle, Rectangle, Arc, Ellipse

import numpy as np

# sound_path = '/Users/nannan/Desktop/librosa/00314.wav'
audio_fea = []

sounddir = 'D:\\Dataset\\audio_processing'
soundname = '1-12654-A-15.wav'
soundpath = os.path.join(sounddir, soundname)

def rec_cal(index_width, index_height, feature_width, feature_height, timemax):
    feature_height_max = 8192
    min_f0, max_f0 = (index_width-1) * (timemax / feature_width), index_width * (timemax / feature_width)
    min_f1, max_f1 = (index_height-1) * (feature_height_max / feature_height), index_height * (feature_height_max / feature_height)
    width = max_f0 - min_f0
    height = max_f1 - min_f1

    return min_f0, min_f1, width, height

def drawgraph(path, index_width, index_height, feature_width=13, feature_height=4):
    wav, sr = librosa.load(soundpath, sr=16000)

    timemax = librosa.get_duration(y=wav, sr=16000, S=None, n_fft=2048, center=True, filename=None)

    melspec = librosa.feature.melspectrogram(wav, sr, n_fft=1024, hop_length=512, n_mels=64) #n_mels 表示频域
    # convert to log scale
    logmelspec = librosa.power_to_db(melspec)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')

    margin = .1

    index_width = [4, 1, 3, 8, 12, 6, 7]
    index_height = [3, 2, 2, 0, 1, 2, 3]
    pic_elements = []

    for i in range(len(index_width)):
        min_f0, min_f1, width, height = rec_cal(index_width[i], index_height[i], feature_width, feature_height, timemax)
        rectangle = patches.Rectangle(
            xy=(min_f0, min_f1),    # point of origin.
            width=width,
            height=height,
            linewidth=1,
            color='green',
            fill=False
        )
        pic_elements.append(rectangle)

    # List of elements to be plotted
    for element in pic_elements:
        ax.add_patch(element)

    plt.show()

drawgraph(soundpath, 5,2)
