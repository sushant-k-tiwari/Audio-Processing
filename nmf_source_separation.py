import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf

audio_file = 'test.wav'

sample_rate = 5512
# Load audio signal
audio_sound, sr = librosa.load(audio_file, sr=sample_rate)
# Display audio
ipd.Audio(audio_sound, rate=sr)

# Plotting the sound's waveform
fig, ax = plt.subplots(figsize=(10, 3))
librosa.display.waveshow(audio_sound, sr=sr, ax=ax, x_axis='time')
ax.set(title='The sound waveform', xlabel='Time [s]')
ax.legend()

FRAME = 512
HOP = 256

# Return the complex Short Term Fourier Transform
sound_stft = librosa.stft(audio_sound, n_fft=FRAME, hop_length=HOP)

# Magnitude Spectrogram
sound_stft_Magnitude = np.abs(sound_stft)

# Phase spectrogram
sound_stft_Angle = np.angle(sound_stft)

# Plot Spectogram
Spec = librosa.amplitude_to_db(sound_stft_Magnitude, ref=np.max)
librosa.display.specshow(Spec, y_axis='hz', sr=sr, hop_length=HOP, x_axis='time', cmap=matplotlib.cm.jet)
plt.title('Audio spectrogram')

epsilon = 1e-10  # error to introduce
V = sound_stft_Magnitude + epsilon
K, N = np.shape(V)
S = 4  # Number of Sources to separate

print(f"S = {S} : Number of Sources to separate")


def divergence(V, W, H, beta=1):
    """
    beta = 2 : Euclidean cost function
    beta = 1 : Kullback-Leibler cost function
    beta = 0 : Itakura-Saito cost function
    """
    if beta == 0:
        return np.sum(V / (W @ H) - np.log10(V / (W @ H)) - 1)

    if beta == 1:
        return np.sum(V * np.log10(V / (W @ H)) + (W @ H - V))

    if beta == 2:
        return 1 / 2 * np.linalg.norm(W @ H - V)


def plot_NMF_iter(W, H, beta, iteration=None):
    f = plt.figure(figsize=(4, 4))
    f.suptitle(f"NMF Iteration {iteration}, for beta = {beta}", fontsize=8,)

    # definitions for the axes
    V_plot = plt.axes([0.35, 0.1, 1, 0.6])
    H_plot = plt.axes([0.35, 0.75, 1, 0.15])
    W_plot = plt.axes([0.1, 0.1, 0.2, 0.6])

    D = librosa.amplitude_to_db(W @ H, ref=np.max)

    librosa.display.specshow(W, y_axis='hz', sr=sr, hop_length=HOP, x_axis='time', cmap=matplotlib.cm.jet, ax=W_plot)
    librosa.display.specshow(H, y_axis='hz', sr=sr, hop_length=HOP, x_axis='time', cmap=matplotlib.cm.jet, ax=H_plot)
    librosa.display.specshow(D, y_axis='hz', sr=sr, hop_length=HOP, x_axis='time', cmap=matplotlib.cm.jet, ax=V_plot)

    W_plot.set_title('Dictionnary W', fontsize=10)
    H_plot.set_title('Temporal activations H', fontsize=10)

    W_plot.axes.get_xaxis().set_visible(False)
    H_plot.axes.get_xaxis().set_visible(False)
    V_plot.axes.get_yaxis().set_visible(False)


def NMF(V, S, beta=1, threshold=0.05, MAXITER=10000, display=True, displayEveryNiter=None):
    counter = 0
    cost_function = []
    beta_divergence = 1

    K, N = np.shape(V)

    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale=2.5, size=(K, S)))
    H = np.abs(np.random.normal(loc=0, scale=2.5, size=(S, N)))

    # Plotting the first initialization
    if display == True:
        plot_NMF_iter(W, H, beta, counter)

    while beta_divergence >= threshold and counter <= MAXITER:

        # Update of W and H
        H *= (W.T @ (((W @ H) ** (beta - 2)) * V)) / (W.T @ ((W @ H) ** (beta - 1)) + 10e-10)
        W *= (((W @ H) ** (beta - 2) * V) @ H.T) / ((W @ H) ** (beta - 1) @ H.T + 10e-10)

        # Compute cost function
        beta_divergence = divergence(V, W, H, beta=1)
        cost_function.append(beta_divergence)

        #if display == True and counter % displayEveryNiter == 0:
            #plot_NMF_iter(W, H, beta, counter)

        counter += 1

    if counter - 1 == MAXITER:
        print(f"Stop after {MAXITER} iterations.")
    else:
        print(f"Convergeance after {counter - 1} iterations.")

    return W, H, cost_function


beta = 1
W, H, cost_function = NMF(V, S, beta=beta, threshold=0.05, MAXITER=10000, display=True, displayEveryNiter=None)

# Plot the cost function
plt.figure(figsize=(5, 3))
plt.plot(cost_function)
plt.title("Cost Function")
plt.xlabel("Number of iteration")
plt.ylabel(f"Beta Divergence for beta = {beta} ")

# After NMF, each audio source S can be expressed as a frequency mask over time
f, axs = plt.subplots(nrows=1, ncols=S, figsize=(20, 5))
filtered_spectrograms = []
for i in range(S):
    axs[i].set_title(f"Frequency Mask of Audio Source s = {i+1}")
    # Filter each source component
    WsHs = W[:, [i]] @ H[[i], :]
    filtered_spectrogram = W[:, [i]] @ H[[i], :] / (W @ H) * V
    # Compute the filtered spectrogram
    D = librosa.amplitude_to_db(filtered_spectrogram, ref=np.max)
    # Show the filtered spectrogram
    #librosa.display.specshow(D, y_axis='hz', sr=sr, hop_length=HOP, x_axis='time', cmap=matplotlib.cm.jet,ax=axs[i])

    filtered_spectrograms.append(filtered_spectrogram)

reconstructed_sounds = []
for i in range(S):
    reconstruct = filtered_spectrograms[i] * np.exp(1j * sound_stft_Angle)
    new_sound = librosa.istft(reconstruct, n_fft=FRAME, hop_length=HOP)
    reconstructed_sounds.append(new_sound)

for i in range(S):
    ipd.display(f"Source {i+1}", ipd.Audio(reconstructed_sounds[i], rate=sr))

# Save the reconstructed sounds
output_files = []
for i in range(S):
    output_file = f'sound_source_{i+1}.wav'
    sf.write(output_file, reconstructed_sounds[i], sr)
    output_files.append(output_file)
    print(f"Source {i+1} saved as {output_file}")

# Display the saved files
print("Reconstructed sounds saved successfully:")
for file in output_files:
    print(file)



