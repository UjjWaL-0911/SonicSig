from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
from recognizer import recognize_song
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATABASE_FOLDER = os.path.join(UPLOAD_FOLDER, 'database')
QUERY_FOLDER = os.path.join(UPLOAD_FOLDER, 'queries')
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Create upload directories if they don't exist
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(QUERY_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
app.config['QUERY_FOLDER'] = QUERY_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_yamnet_embeddings(file_path, target_sr=16000):
    """Extract embeddings using YAMNet."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

def compare_yamnet_embeddings(emb1, emb2):
    """Compare YAMNet embeddings using cosine similarity."""
    avg_emb1 = emb1.mean(axis=0).reshape(1, -1)
    avg_emb2 = emb2.mean(axis=0).reshape(1, -1)
    return cosine_similarity(avg_emb1, avg_emb2)[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['QUERY_FOLDER'], filename)
            file.save(filepath)
            
            # Perform fingerprint-based recognition
            fingerprint_result = recognize_song(filepath)
            
            # Perform embedding-based recognition
            query_embeddings = extract_yamnet_embeddings(filepath)
            embedding_results = []
            for song_file in os.listdir(app.config['DATABASE_FOLDER']):
                if allowed_file(song_file):
                    song_path = os.path.join(app.config['DATABASE_FOLDER'], song_file)
                    song_embeddings = extract_yamnet_embeddings(song_path)
                    similarity = compare_yamnet_embeddings(query_embeddings, song_embeddings)
                    embedding_results.append((song_file, similarity))
            
            # Find the best match from embeddings
            embedding_results.sort(key=lambda x: x[1], reverse=True)
            best_embedding_match = embedding_results[0] if embedding_results else None
            
            # Combine results
            if fingerprint_result['match_found'] and best_embedding_match:
                # Convert confidence to float (handle both '95%' and float cases)
                try:
                    fingerprint_conf = float(str(fingerprint_result['confidence']).strip('%'))
                except Exception:
                    fingerprint_conf = 0.0
                embedding_conf = best_embedding_match[1] * 100  # Convert to percentage for fair comparison

                combined_result = {
                    'match_found': True,
                    'song_name': fingerprint_result['song_name'] if fingerprint_conf > embedding_conf else best_embedding_match[0],
                    'confidence': max(fingerprint_conf, embedding_conf),
                    'matches': fingerprint_result['matches']
                }
            elif fingerprint_result['match_found']:
                combined_result = fingerprint_result
            elif best_embedding_match:
                combined_result = {
                    'match_found': True,
                    'song_name': best_embedding_match[0],
                    'confidence': best_embedding_match[1] * 100,
                    'matches': 0
                }
            else:
                combined_result = {
                    'match_found': False,
                    'message': 'No match found'
                }
            
            # Delete the query file after processing
            os.remove(filepath)
            
            return render_template('result.html', result=combined_result)
    
    # Get list of songs in database
    database_songs = []
    for file in os.listdir(app.config['DATABASE_FOLDER']):
        if allowed_file(file):
            database_songs.append(file)
    
    return render_template('index.html', database_songs=database_songs)

@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    if 'reference_file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    
    file = request.files['reference_file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['DATABASE_FOLDER'], filename)
        file.save(filepath)
        flash(f'Successfully added {filename} to database')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)