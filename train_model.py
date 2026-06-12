import os
import numpy as np
import librosa
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, 'cat_dataset')
SAVE_PATH = os.path.join(BASE_DIR, 'backend', 'model.h5')

CLASSES  = ['angry', 'fighting', 'happy', 'sad']
SR       = 22050
DURATION = 4      # seconds per clip (pad/trim to this)
N_MFCC   = 40
HOP      = 512


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — returns (n_mfcc, time_frames) for each clip
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(y, sr):
    y, _ = librosa.effects.trim(y)
    # Pad or trim to fixed duration
    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    y = librosa.util.normalize(y)

    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    delta     = librosa.feature.delta(mfcc)
    delta2    = librosa.feature.delta(mfcc, order=2)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=HOP)
    rms       = librosa.feature.rms(y=y, hop_length=HOP)

    # Stack all features: shape (features, time)
    feat = np.vstack([mfcc, delta, delta2, chroma, zcr, rms])
    return feat.T   # → (time, features)


def augment(y, sr, method):
    if method == 'noise':
        return y + np.random.randn(len(y)) * 0.004
    elif method == 'pitch_up':
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    elif method == 'pitch_down':
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    elif method == 'speed_up':
        return librosa.effects.time_stretch(y, rate=1.1)
    elif method == 'speed_down':
        return librosa.effects.time_stretch(y, rate=0.9)
    elif method == 'shift':
        return np.roll(y, int(sr * 0.3))
    return y

AUGMENT_METHODS = ['noise', 'pitch_up', 'pitch_down', 'speed_up', 'speed_down', 'shift']


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load audio + extract features (with augmentation)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading audio and extracting features")
print("=" * 60)

X, y_labels = [], []

for cls in CLASSES:
    folder = os.path.join(AUDIO_DIR, cls)
    files  = [f for f in os.listdir(folder)
              if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    print(f"\n  {cls}: {len(files)} files")
    loaded = 0

    for fn in files:
        try:
            y, sr = librosa.load(os.path.join(folder, fn), sr=SR)
            # Original
            feat = extract_features(y, sr)
            X.append(feat)
            y_labels.append(cls)
            # All augmentations
            for method in AUGMENT_METHODS:
                y_aug  = augment(y, sr, method)
                feat   = extract_features(y_aug, sr)
                X.append(feat)
                y_labels.append(cls)
            loaded += 1
        except Exception as e:
            print(f"    SKIP {fn}: {e}")

    print(f"    → {loaded * (1 + len(AUGMENT_METHODS))} feature vectors")

X = np.array(X)
print(f"\nDataset shape: {X.shape}")   # (samples, time, features)
print("Class dist:", Counter(y_labels))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Encode labels + train/val split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Encoding labels and splitting data")
print("=" * 60)

le      = LabelEncoder()
le.fit(CLASSES)   # fixed order: angry=0, fighting=1, happy=2, sad=3
y_enc   = le.transform(y_labels)
y_cat   = to_categorical(y_enc, num_classes=4)

print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build Conv1D model on audio features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Building and training model")
print("=" * 60)

n_timesteps = X.shape[1]
n_features  = X.shape[2]

model = Sequential([
    Input(shape=(n_timesteps, n_features)),

    Conv1D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 3, activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=6, min_lr=1e-6, verbose=1),
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print(f"\nModel saved to: {SAVE_PATH}")

# Save label order so app.py uses same mapping
label_order_path = os.path.join(BASE_DIR, 'backend', 'label_order.txt')
with open(label_order_path, 'w') as f:
    for cls in le.classes_:
        f.write(cls + '\n')
print(f"Label order saved to: {label_order_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Validation report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Validation Report")
print("=" * 60)

preds    = model.predict(X_val, verbose=0)
pred_cls = np.argmax(preds, axis=1)
true_cls = np.argmax(y_val, axis=1)

print("True dist:", Counter([le.classes_[t] for t in true_cls]))
print("Pred dist:", Counter([le.classes_[p] for p in pred_cls]))
print()
print(classification_report(true_cls, pred_cls, target_names=le.classes_))