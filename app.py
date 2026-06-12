import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from tensorflow.keras.models import load_model
import os
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model(os.path.join(BASE_DIR, "model.h5"))

# Load label order
label_order_path = os.path.join(BASE_DIR, "label_order.txt")
if os.path.exists(label_order_path):
    with open(label_order_path) as f:
        CLASSES = [line.strip() for line in f if line.strip()]
else:
    CLASSES = ['angry', 'fighting', 'happy', 'sad']

EMOTION_MAP = {
    "angry":    "😾 Angry",
    "fighting": "⚔️ Fighting",
    "happy":    "😸 Happy",
    "sad":      "😿 Sad"
}

UPLOAD_FOLDER      = os.path.join(BASE_DIR, "static", "uploads")
# spectrogram_dataset lives at the project root (one level above backend/)
PROJECT_ROOT       = os.path.dirname(BASE_DIR)
SPECTROGRAM_FOLDER = os.path.join(PROJECT_ROOT, "spectrogram_dataset")

# Ensure per-emotion subfolders exist
for cls in ['angry', 'fighting', 'happy', 'sad']:
    os.makedirs(os.path.join(SPECTROGRAM_FOLDER, cls), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SR       = 22050
DURATION = 4
N_MFCC   = 40
HOP      = 512


def extract_features(y, sr):
    y, _ = librosa.effects.trim(y)
    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    y = librosa.util.normalize(y)

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    zcr    = librosa.feature.zero_crossing_rate(y, hop_length=HOP)
    rms    = librosa.feature.rms(y=y, hop_length=HOP)

    feat = np.vstack([mfcc, delta, delta2, chroma, zcr, rms])
    return feat.T


def save_spectrogram(y, sr, emotion_key, filename_stem):
    """
    Generate a mel-spectrogram image, save it into
    static/spectrograms/<emotion>/<stem>.png
    Returns the web-relative path  (for url_for / jsonify).
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=HOP,
                                   x_axis='time', y_axis='mel', ax=ax,
                                   cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(f'Mel-Spectrogram — {emotion_key}')
    plt.tight_layout()

    spec_filename = f"{filename_stem}.png"
    spec_path = os.path.join(SPECTROGRAM_FOLDER, emotion_key, spec_filename)
    fig.savefig(spec_path)
    plt.close(fig)

    # Return path relative to project root spectrogram_dataset
    return f"spectrogram_dataset/{emotion_key}/{spec_filename}"


def run_prediction(file_path, filename_stem=None):
    y, sr = librosa.load(file_path, sr=SR)

    # Pad/trim for features
    y_feat, _ = librosa.effects.trim(y)
    target_len = SR * DURATION
    if len(y_feat) < target_len:
        y_feat = np.pad(y_feat, (0, target_len - len(y_feat)))
    else:
        y_feat = y_feat[:target_len]
    y_feat = librosa.util.normalize(y_feat)

    feat = extract_features(y, sr)
    feat = np.expand_dims(feat, axis=0)

    pred = model.predict(feat, verbose=0)[0]
    idx  = int(np.argmax(pred))
    emotion_key = CLASSES[idx]

    # Save spectrogram into the correct emotion subfolder
    stem = filename_stem or uuid.uuid4().hex
    spec_rel_path = save_spectrogram(y_feat, SR, emotion_key, stem)

    return {
        "prediction":  EMOTION_MAP[emotion_key],
        "confidence":  round(float(pred[idx]) * 100, 2),
        "spectrogram": spec_rel_path,   # e.g. "spectrograms/angry/abc123.png"
        "all_scores": {
            EMOTION_MAP[CLASSES[i]]: round(float(pred[i]) * 100, 2)
            for i in range(len(CLASSES))
        }
    }


def webm_to_wav(raw_path, wav_path):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_frame_rate(SR).set_channels(1)
        audio.export(wav_path, format="wav")
        return
    except Exception:
        pass

    try:
        data, rate = sf.read(raw_path, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        sf.write(wav_path, data, rate, subtype='PCM_16')
        return
    except Exception:
        pass

    raise RuntimeError("Could not convert browser audio. Run: pip install pydub")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = confidence = audio_file = error = all_scores = spectrogram = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            safe_name = os.path.basename(file.filename)
            stem      = os.path.splitext(safe_name)[0]
            file_path = os.path.join(UPLOAD_FOLDER, safe_name)
            file.save(file_path)
            try:
                result      = run_prediction(file_path, filename_stem=stem)
                prediction  = result["prediction"]
                confidence  = result["confidence"]
                all_scores  = result["all_scores"]
                spectrogram = result["spectrogram"]
                audio_file  = safe_name
            except Exception as e:
                error = f"Prediction failed: {str(e)}"
        else:
            error = "No file uploaded."

    return render_template('index.html',
        prediction=prediction, confidence=confidence,
        audio_file=audio_file, all_scores=all_scores,
        spectrogram=spectrogram, error=error)


@app.route('/predict_live', methods=['POST'])
def predict_live():
    uid      = uuid.uuid4().hex
    raw_path = None
    wav_path = None
    try:
        blob = request.files.get('audio')
        if not blob:
            return jsonify({"error": "No audio received"}), 400

        raw_path = os.path.join(UPLOAD_FOLDER, f"live_{uid}.webm")
        wav_path = os.path.join(UPLOAD_FOLDER, f"live_{uid}.wav")
        blob.save(raw_path)

        webm_to_wav(raw_path, wav_path)
        result = run_prediction(wav_path, filename_stem=f"live_{uid}")
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [raw_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.route('/spectrogram_dataset/<path:filepath>')
def serve_spectrogram(filepath):
    """Serve spectrogram images stored outside the static folder."""
    directory = os.path.join(PROJECT_ROOT, 'spectrogram_dataset')
    return send_from_directory(directory, filepath)


if __name__ == '__main__':
    app.run(debug=True)
