import os
import numpy as np
import random
import torchaudio
from torchaudio import transforms
from keras import models, layers
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import torch

def open_audio(audio_file_path):
    data, sampling_rate = torchaudio.load(audio_file_path)
    return data, sampling_rate

def rechannel_audio(audio, new_channel):
    data, sampling_rate = audio

    if data.shape[0] == new_channel:
        return audio

    if new_channel == 1:
        resig = data[:1, :]
    else:
        resig = torch.cat([data, data])

    return resig, sampling_rate

def resample_audio(audio, new_sampling_rate):
    data, sampling_rate = audio

    if sampling_rate == new_sampling_rate:
        return audio

    num_channels = data.shape[0]
    resig = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(data[:1, :])
    if num_channels > 1:
        retwo = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(data[1:, :])
        resig = torch.cat([resig, retwo])

    return resig, new_sampling_rate

def pad_trunc_audio(audio, max_ms):
    data, sampling_rate = audio
    num_rows, data_len = data.shape
    max_len = sampling_rate // 1000 * max_ms

    if data_len > max_len:
        data = data[:, :max_len]
    elif data_len < max_len:
        pad_begin_len = random.randint(0, max_len - data_len)
        pad_end_len = max_len - data_len - pad_begin_len
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        data = torch.cat((pad_begin, data, pad_end), 1)

    return data, sampling_rate

def spectrogram_audio(audio, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = audio
    top_db = 80
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec

def data_processing(folder):
    spectrograms = []
    new_channel = 2  
    new_sampling_rate = 44100  
    duration = 6000
    
    for i in os.listdir(folder):
        audio = open_audio(os.path.join(folder, i))
        resampled_audio = resample_audio(audio, new_sampling_rate)
        rechanneled_audio = rechannel_audio(resampled_audio, new_channel)
        padded_audio = pad_trunc_audio(rechanneled_audio, duration)
        spectro_gram = spectrogram_audio(padded_audio, n_mels=64, n_fft=1024, hop_len=None)
        spectrograms.append(spectro_gram)

    return spectrograms

violin_spectrograms = data_processing(r"D:\Documents\SEM_6\Project\Data_set\archive\Musical_Instrument_Data\violin_files")  # cls id = 0
violin_arr_list = [i.numpy() for i in violin_spectrograms]
violin_arr = np.array(violin_arr_list)
print(violin_arr.shape)

mohanveena_spectrograms = data_processing(r"D:\Documents\SEM_6\Project\Data_set\archive\Musical_Instrument_Data\mohanveena_files")  # cls id = 1
mohanveena_arr_list = [i.numpy() for i in mohanveena_spectrograms]
mohanveena_arr = np.array(mohanveena_arr_list)
print(mohanveena_arr.shape)

sitar_spectrograms = data_processing(r"D:\Documents\SEM_6\Project\Data_set\archive\Musical_Instrument_Data\sitar_files")  # cls id = 2
sitar_arr_list = [i.numpy() for i in sitar_spectrograms]
sitar_arr = np.array(sitar_arr_list)
print(sitar_arr.shape)

y1 = np.zeros(6)
y2 = np.ones(10)
y3 = np.full(10, 2)
print(y1.shape)
print(y2.shape)
print(y3.shape)
y = np.concatenate((y1, y2, y3), axis=0)
print(y.shape)
y = y.reshape(26, 1)
print(y.shape)

x = np.concatenate((violin_arr, mohanveena_arr, sitar_arr), axis=0)
print(x.shape)

network_model = models.Sequential()
network_model.add(layers.Dense(512, activation="leaky_relu", input_shape=(2 * 64 * 516,)))
network_model.add(layers.Dense(128, activation="relu", input_shape=(2 * 64 * 516,)))
network_model.add(Dropout(0.5))
network_model.add(layers.Dense(3, activation="softmax"))

network_model.summary()

network_model.compile(optimizer="adam", metrics=["accuracy"], loss="categorical_crossentropy")

x = x.reshape(26, 2 * 64 * 516)
x = x.astype(float) / 255  
print(x.shape)
print(y.shape)

y = to_categorical(y)

print(x.shape)
print(y.shape)  

network_model.fit(x, y, epochs=15)


def preprocess_audio(audio_file_path):
    audio_data, sampling_rate = open_audio(audio_file_path)
    audio_data, _ = rechannel_audio((audio_data, sampling_rate), new_channel=2)
    audio_data, _ = resample_audio((audio_data, sampling_rate), new_sampling_rate=44100)
    audio_data, _ = pad_trunc_audio((audio_data, sampling_rate), max_ms=6000)
    spectrogram = spectrogram_audio((audio_data, sampling_rate), n_mels=64, n_fft=1024, hop_len=None)
    return spectrogram


def classify_audio(audio_file_path, model):
    spectrogram = preprocess_audio(audio_file_path)
    spectrogram_np = spectrogram.numpy()

    expected_input_shape = (64, 516)
    if spectrogram_np.shape[2] < expected_input_shape[1]:
        pad_width = expected_input_shape[1] - spectrogram_np.shape[2]
        spectrogram_np = np.pad(spectrogram_np, ((0, 0), (0, 0), (0, pad_width)))
    elif spectrogram_np.shape[2] > expected_input_shape[1]:
        spectrogram_np = spectrogram_np[:, :, :expected_input_shape[1]]

    spectrogram_flat = spectrogram_np.reshape(1, -1).astype(float) / 255

    prediction = model.predict(spectrogram_flat)
    predicted_class = np.argmax(prediction)
    return predicted_class

trained_model = network_model

audio_file_path = r"D:\Documents\SEM_6\Project\sound_source_3.wav"
predicted_class = classify_audio(audio_file_path, trained_model)

def named(score):
    if(score == 1):
        print("Violin")
print(named(predicted_class))
#print("Predicted Class:", predicted_class)
