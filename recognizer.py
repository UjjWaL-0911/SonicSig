import librosa
import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion
import os

def load_audio(file_path, sr=11025):
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate

def compute_spectrogram(audio, n_fft=2048, hop_length=512):
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return stft

def find_peaks(spectrogram, threshold=20):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
    background = (spectrogram == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    peaks = np.where(detected_peaks & (spectrogram > threshold))
    return list(zip(peaks[0], peaks[1]))

def generate_hashes(peaks, fan_value=15):
    fingerprints = []
    for i in range(len(peaks)):
        for j in range(1, min(fan_value, len(peaks) - i)):
            freq1, time1 = peaks[i]
            freq2, time2 = peaks[i + j]
            delta_time = time2 - time1
            if delta_time <= 200:  # Maximum time delta
                hash_str = f"{freq1}|{freq2}|{delta_time}"
                fingerprints.append((hash_str, time1))
    return fingerprints

def recognize_song(query_path, database_path='static/uploads/database'):
    # Process query audio
    query_audio, sr = load_audio(query_path)
    query_spec = compute_spectrogram(query_audio)
    query_peaks = find_peaks(query_spec)
    query_hashes = generate_hashes(query_peaks)
    
    best_match = None
    max_matches = 0
    
    # Compare with each song in database
    for song_file in os.listdir(database_path):
        if not song_file.endswith(('.mp3', '.wav')):
            continue
            
        song_path = os.path.join(database_path, song_file)
        song_audio, _ = load_audio(song_path)
        song_spec = compute_spectrogram(song_audio)
        song_peaks = find_peaks(song_spec)
        song_hashes = generate_hashes(song_peaks)
        
        # Convert song hashes to dictionary for faster lookup
        song_hash_dict = {}
        for h, t in song_hashes:
            if h not in song_hash_dict:
                song_hash_dict[h] = []
            song_hash_dict[h].append(t)
        
        # Count matches
        matches = 0
        for query_hash, _ in query_hashes:
            if query_hash in song_hash_dict:
                matches += 1
        
        if matches > max_matches:
            max_matches = matches
            best_match = song_file
    
    if best_match and max_matches > 10:  # Minimum threshold for a match
        confidence = (max_matches / len(query_hashes)) * 100
        return {
            'match_found': True,
            'song_name': best_match,
            'confidence': f"{confidence:.2f}%",
            'matches': max_matches
        }
    else:
        return {
            'match_found': False,
            'message': 'No match found'
        }