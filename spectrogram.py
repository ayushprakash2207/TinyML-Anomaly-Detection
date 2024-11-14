import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load normal and abnormal audio files
normal_audio_path = "Dataset/data/normal/Fan_4.wav"
abnormal_audio_path = "Dataset/data/abnormal/Fan_Bad_4.wav"

# Load audio data
normal_audio, sr = librosa.load(normal_audio_path)
abnormal_audio, sr = librosa.load(abnormal_audio_path)

# Extract Mel spectrograms
normal_mel = librosa.feature.melspectrogram(y=normal_audio, sr=sr)
abnormal_mel = librosa.feature.melspectrogram(y=abnormal_audio, sr=sr)

# Convert to dB scale
normal_mel_db = librosa.power_to_db(normal_mel, ref=np.max)
abnormal_mel_db = librosa.power_to_db(abnormal_mel, ref=np.max)

# Display spectrograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
librosa.display.specshow(normal_mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.title('Normal Audio Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.subplot(1, 2, 2)
librosa.display.specshow(abnormal_mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.title('Abnormal Audio Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()