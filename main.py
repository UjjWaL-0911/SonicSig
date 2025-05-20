import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf

#-------------------------------
# Fingerprint-Based Functions (Traditional Shazam Approach)
#-------------------------------
def load_audio(file_path, sr=11025):
    """Load an audio file using librosa."""
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate

def compute_spectrogram(audio, n_fft=2048, hop_length=512):
    """Compute the magnitude spectrogram from audio signal."""
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return stft

def plot_waveform(audio, sr):
    """Plot the waveform of the audio signal."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(stft, sr, hop_length):
    """Plot the spectrogram."""
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                             sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.title('Spectrogram (Time vs Frequency)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def find_peaks(spectrogram, size=20, threshold=20):
    """Find local peaks in the spectrogram as candidate fingerprints."""
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(spectrogram, footprint=neighborhood, size=size) == spectrogram
    background = (spectrogram == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    peaks = np.where(detected_peaks & (spectrogram > threshold))
    return peaks

def plot_peaks_on_spectrogram(stft, sr, hop_length, peaks):
    """Display peaks on the spectrogram."""
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                             sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.scatter(peaks[1] * hop_length / sr, peaks[0], color='green', marker='x')
    plt.title('Spectrogram with Peaks')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def generate_hashes(peaks, fan_value=15):
    """Generate fingerprints based on peak pairs."""
    fingerprints = []
    peak_freqs = peaks[0]
    peak_times = peaks[1]
    for i in range(len(peak_freqs)):
        for j in range(1, fan_value):
            if i + j < len(peak_freqs):
                freq1 = peak_freqs[i]
                freq2 = peak_freqs[i + j]
                t1 = peak_times[i]
                t2 = peak_times[i + j]
                delta_t = t2 - t1
                if 0 < delta_t <= 200:
                    h = hash((freq1, freq2, delta_t))
                    fingerprints.append((h, t1))
    return fingerprints

def build_database(fingerprints):
    """Create a fingerprint database for a reference audio file."""
    database = {}
    for h, t in fingerprints:
        if h not in database:
            database[h] = []
        database[h].append(t)
    return database

def match_fingerprints(query_fingerprints, database):
    """Match query fingerprints against the database."""
    offset_counter = {}
    for h, qt in query_fingerprints:
        if h in database:
            for dt in database[h]:
                offset = dt - qt
                offset_counter[offset] = offset_counter.get(offset, 0) + 1
    if not offset_counter:
        return None, 0
    best_offset = max(offset_counter, key=offset_counter.get)
    return best_offset, offset_counter[best_offset]

#-------------------------------
# Embedding-Based Functions using YAMNet
#-------------------------------
def extract_yamnet_embeddings(file_path, target_sr=16000):
    """
    Extract embeddings using YAMNet.
    YAMNet expects the waveform sampled at 16kHz.
    Returns the embeddings along with scores and spectrogram.
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    # Convert audio waveform to a tensor of type float32
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    # Load YAMNet model from TensorFlow Hub
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    # Run YAMNet. Note: Model expects waveform shape [n_samples].
    scores, embeddings, spectrogram = yamnet_model(waveform)
    # embeddings has shape [frames, 1024]
    return embeddings.numpy(), scores.numpy(), spectrogram.numpy()

def compare_yamnet_embeddings(emb1, emb2):
    """
    Compare YAMNet embeddings by averaging over time and computing cosine similarity.
    """
    avg_emb1 = np.mean(emb1, axis=0).reshape(1, -1)
    avg_emb2 = np.mean(emb2, axis=0).reshape(1, -1)
    sim = cosine_similarity(avg_emb1, avg_emb2)[0][0]
    return sim

#-------------------------------
# Main Execution: Both Approaches Demonstrated
#-------------------------------
if __name__ == '__main__':
    # ----- Example 1: Fingerprint Based Matching -----
    print("=== Fingerprint Based Matching ===")
    ref_file = "shape_of_you.mp3"  # Change path as necessary
    sr_fp = 11025
    audio, sr_loaded = load_audio(ref_file, sr=sr_fp)
    
    # Compute and display spectrogram/waveform
    spectrogram = compute_spectrogram(audio)
    plot_waveform(audio, sr_loaded)
    plot_spectrogram(spectrogram, sr_loaded, hop_length=512)
    
    # Find peaks and display them
    peaks = find_peaks(spectrogram, size=20, threshold=20)
    plot_peaks_on_spectrogram(spectrogram, sr_loaded, hop_length=512, peaks=peaks)
    
    # Generate fingerprints and build the database from the reference track
    fingerprints = generate_hashes(peaks)
    database = build_database(fingerprints)
    
    # Load query audio and perform matching
    query_file_fp = "shapeOfYou_recorded.wav"  # Change path as needed
    query_audio, _ = load_audio(query_file_fp, sr=sr_fp)
    query_spectrogram = compute_spectrogram(query_audio)
    query_peaks = find_peaks(query_spectrogram, size=20, threshold=20)
    query_fingerprints = generate_hashes(query_peaks)
    offset, match_count = match_fingerprints(query_fingerprints, database)
    
    if match_count > 0:
        print(f"Fingerprint match found! Time offset: {offset} (matches: {match_count})")
    else:
        print("No fingerprint match found.")
    
    # ----- Example 2: Embedding Based Matching with YAMNet -----
    print("\n=== Embedding Based Matching with YAMNet ===")
    # Extract embeddings for the reference audio (YAMNet requires a 16kHz sample rate)
    ref_file_yamnet = "shape_of_you.mp3"  # Change path as necessary
    emb_ref, scores_ref, spec_ref = extract_yamnet_embeddings(ref_file_yamnet, target_sr=16000)
    print("Reference YAMNet embeddings shape:", emb_ref.shape)
    
    # Extract embeddings for the query audio
    query_file_yamnet = "shapeOfYou_recorded.wav"  # Change path as needed
    emb_query, scores_query, spec_query = extract_yamnet_embeddings(query_file_yamnet, target_sr=16000)
    print("Query YAMNet embeddings shape:", emb_query.shape)
    
    # Compare embeddings using cosine similarity (average over time frames)
    sim_score = compare_yamnet_embeddings(emb_ref, emb_query)
    print(f"Cosine similarity between reference and query YAMNet embeddings: {sim_score:.4f}")
    
    # Optionally, save a clip from the query audio using soundfile
    query_audio_sf, sr_sf = load_audio(query_file_yamnet, sr=sr_fp)
    sf.write("shape_of_you_clip.wav", query_audio_sf, sr_sf)
    
    print("Done.")