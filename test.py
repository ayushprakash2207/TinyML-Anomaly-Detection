import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define paths to folders containing normal and abnormal audio files
normal_audio_folder = 'Dataset/data/normal'  # Replace with your folder path
anomaly_audio_folder = 'Dataset/data/abnormal'  # Replace with your folder path

# Set parameters
n_mels = 64
hop_length = 512
segment_length = 128

# Convert audio to Mel spectrograms
def audio_to_mel_spectrogram(audio, sr, n_mels=64, hop_length=512):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# Function to load and process audio files into Mel spectrograms
def load_and_preprocess_audio(audio_folder):
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    mel_spectrograms = []
    for file in audio_files:
        audio_path = os.path.join(audio_folder, file)
        audio, sr = librosa.load(audio_path, sr=None)
        mel_spec = audio_to_mel_spectrogram(audio, sr, n_mels, hop_length)
        mel_spectrograms.append(mel_spec)
    return mel_spectrograms, audio_files  # Return the file names as well

# Load and preprocess audio
normal_mel_specs, normal_audio_files = load_and_preprocess_audio(normal_audio_folder)
anomaly_mel_specs, anomaly_audio_files = load_and_preprocess_audio(anomaly_audio_folder)

# Function to reshape spectrograms into segments
def reshape_spectrogram(spectrogram, segment_length=128):
    segments = []
    for i in range(0, spectrogram.shape[1] - segment_length, segment_length):
        segment = spectrogram[:, i:i + segment_length]
        segments.append(segment)
    return np.array(segments)

# Reshape spectrograms into segments, keeping track of file names
normal_segments = []
normal_segment_filenames = []
for i, spec in enumerate(normal_mel_specs):
    segments = reshape_spectrogram(spec, segment_length)
    normal_segments.extend(segments)
    normal_segment_filenames.extend([normal_audio_files[i]] * len(segments))

anomaly_segments = []
anomaly_segment_filenames = []
for i, spec in enumerate(anomaly_mel_specs):
    segments = reshape_spectrogram(spec, segment_length)
    anomaly_segments.extend(segments)
    anomaly_segment_filenames.extend([anomaly_audio_files[i]] * len(segments))

# Reshape for model input
normal_segments = np.array(normal_segments)[..., np.newaxis]
anomaly_segments = np.array(anomaly_segments)[..., np.newaxis]

print("Normal audio segments shape:", normal_segments.shape)
print("Anomaly audio segments shape:", anomaly_segments.shape)

# Define the model save path
model_save_path = "autoencoder_model.h5"

# Build the autoencoder model with explicit MeanSquaredError loss
def build_autoencoder(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())  # Use explicit MeanSquaredError loss
    return model

# Check if model exists, otherwise build and train
if os.path.exists(model_save_path):
    # Load the model if it exists
    autoencoder = tf.keras.models.load_model(model_save_path)
    print("Loaded model from disk.")
else:
    # Build and train the model if it doesn't exist
    autoencoder = build_autoencoder(normal_segments.shape[1:])
    autoencoder.summary()
    
    # Train the autoencoder
    history = autoencoder.fit(
        normal_segments, normal_segments,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )
    
    # Save the model
    autoencoder.save(model_save_path)
    print(f"Model saved at {model_save_path}")

# Evaluate and detect anomalies
# Calculate reconstruction error on anomaly segments
reconstructed = autoencoder.predict(anomaly_segments)
mse = np.mean(np.power(anomaly_segments - reconstructed, 2), axis=(1, 2, 3))

# Set a threshold for anomaly detection
threshold = np.mean(mse) + 0.1 * np.std(mse)

# Find anomalous segments and their corresponding file names
anomalies = mse > threshold
anomaly_filenames = [anomaly_segment_filenames[i] for i in range(len(anomaly_segments)) if anomalies[i]]

print("Anomaly detected in files:", anomaly_filenames)

# Calculate reconstruction error on normal segments
normal_reconstructed = autoencoder.predict(normal_segments)
normal_mse = np.mean(np.power(normal_segments - normal_reconstructed, 2), axis=(1, 2, 3))

# Plot the distribution of MSE
plt.figure(figsize=(8, 6))
plt.hist(normal_mse, bins=30, alpha=0.5, label='Normal')
plt.hist(mse, bins=30, alpha=0.5, label='Abnormal')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')
plt.legend()
plt.show()