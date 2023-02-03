import librosa
import numpy as np
import matplotlib.pyplot as plt

# load the audio file
filename = "/home/tyler/Documents/Python Projects/usanationalanthem.mp3"
y, sr = librosa.load(filename)

# perform short-time Fourier transform (STFT)
stft = np.abs(librosa.stft(y))

# calculate the power spectral density (PSD)
psd = np.abs(stft)**2

# plot the PSD
plt.figure()
plt.imshow(np.log(psd), origin='lower', aspect='auto', cmap='viridis')
plt.title("Power Spectral Density")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.colorbar()

# get the mel-scaled spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# plot the mel-scaled spectrogram
plt.figure()
plt.imshow(np.log(mel_spectrogram), origin='lower', aspect='auto', cmap='viridis')
plt.title("Mel-scaled Spectrogram")
plt.xlabel("Frame")
plt.ylabel("Mel-frequency")
plt.colorbar()

# calculate the chromagram
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)

# plot the chromagram
plt.figure()
plt.imshow(chromagram, origin='lower', aspect='auto', cmap='viridis')
plt.title("Chromagram")
plt.xlabel("Frame")
plt.ylabel("Chroma")
plt.colorbar()

# show the plots
plt.show()
