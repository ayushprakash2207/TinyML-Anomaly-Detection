import librosa
import soundfile as sf

def split_audio(audio_file, num_parts, start_number=1):
    """
    Splits an audio file into equal parts.

    Args:
        audio_file (str): Path to the audio file.
        num_parts (int): Number of parts to split the audio into.
        start_number (int): The number to start the file names from.

    Returns:
        None
    """

    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Calculate the duration of each part
    part_duration = len(y) / sr / num_parts

    # Split the audio into parts
    for i in range(num_parts):
        start_sample = int(i * part_duration * sr)
        end_sample = int((i + 1) * part_duration * sr)

        # Extract the part of the audio
        part = y[start_sample:end_sample]

        # Save the part as a new audio file with user-specified starting number
        sf.write(f"{audio_file[:-4]}_{start_number + i}.wav", part, sr)

# Example usage:
audio_file = "Raw_Data\Fan_Bad.wav"  # Replace with your audio file path
num_parts = 30  # Number of parts to split into
start_number = 26  # Start the file names from 5

split_audio(audio_file, num_parts, start_number)