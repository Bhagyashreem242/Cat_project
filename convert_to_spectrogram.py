import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "cat_dataset"
SAVE_PATH = "spectrogram_dataset"

os.makedirs(SAVE_PATH, exist_ok=True)

TOTAL = 0

for emotion in os.listdir(DATASET_PATH):

    emotion_path = os.path.join(DATASET_PATH, emotion)

    if not os.path.isdir(emotion_path):
        continue

    save_emotion_path = os.path.join(SAVE_PATH, emotion)

    os.makedirs(save_emotion_path, exist_ok=True)

    count = 0

    for audio in os.listdir(emotion_path):

        audio_path = os.path.join(emotion_path, audio)

        try:

            y, sr = librosa.load(
                audio_path,
                sr=22050
            )

            # Trim silence
            y, _ = librosa.effects.trim(y)

            # Normalize
            y = librosa.util.normalize(y)

            # Add slight noise augmentation
            noise = np.random.randn(len(y))
            y = y + 0.005 * noise

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=256,
                hop_length=512,
                fmax=10000
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            plt.figure(figsize=(4,4))

            librosa.display.specshow(
                mel_db,
                sr=sr,
                cmap='inferno'
            )

            plt.axis('off')

            file_name = os.path.splitext(audio)[0] + ".png"

            save_path = os.path.join(
                save_emotion_path,
                file_name
            )

            plt.savefig(
                save_path,
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close()

            count += 1
            TOTAL += 1

            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Error: {audio} -> {e}")

    print(f"\n✅ {emotion}: {count}")

print(f"\n🎉 TOTAL: {TOTAL}")