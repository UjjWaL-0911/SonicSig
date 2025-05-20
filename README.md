# Sonic Sig

## Introduction

This project is a web-based **song recognition system** built with Flask. Users can upload an audio clip, and the app will attempt to identify the song by comparing it against a database of reference tracks. The system combines classic audio fingerprinting (Shazam-style) and modern deep learning (YAMNet embeddings) for robust and accurate recognition. You can also add new reference songs to the database via the web interface.

---

## Setup and Running

### 1. Clone the Repository

```sh
git clone https://github.com/UjjWaL-0911/SonicSig.git
```

### 2. Create and Activate a Virtual Environment

```sh
python -m venv venv1
venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Set the Flask Secret Key (Recommended for Production)

Generate a secret key:

```sh
python -c "import secrets; print(secrets.token_hex(32))"
```

Set it as an environment variable:

```sh
set FLASK_SECRET_KEY=your_generated_key   # Command Prompt
# or
$env:FLASK_SECRET_KEY="your_generated_key"   # PowerShell
```

### 5. Run the App

```sh
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## How It Works

### Audio Processing & Recognition

- **Fingerprint-Based Recognition:**  
  The app computes a spectrogram of the audio, detects prominent frequency peaks, and generates unique hashes (fingerprints) from pairs of peaks. These fingerprints are matched against those of reference songs to find the best match. This approach is robust to noise and short clips.

- **Embedding-Based Recognition (YAMNet):**  
  The app uses [YAMNet](https://tfhub.dev/google/yamnet/1) (a deep learning model) to extract high-level audio embeddings from both the query and reference tracks. It then computes cosine similarity between embeddings to identify the closest match. This method captures more abstract audio features and is robust to variations in timbre and instrumentation.

- **Result Combination:**  
  Both methods are run for each query. The result with the higher confidence is shown to the user.

### Features

- Upload an audio clip to identify the song.
- Add new reference songs to the database.
- Supports `.mp3` and `.wav` files.
- Uses both classic signal processing and deep learning for robust recognition.

---

## Project Structure

```
Song Recognizer App/
│
├── static/uploads/         # Uploaded and reference audio files
├── templates/              # HTML templates
├── app_secrets.py             
├── app.py                  # Main Flask app
├── recognizer.py           # Fingerprint recognition logic
├── main.py                 #Fingerprint + Embedding Based Recognition Logic
├── README.md
└── requirements.txt
```

---

## Notes

- For best results, use clear audio clips with minimal background noise.
- The app is for educational/demo purposes and can be extended for larger-scale use.

---
