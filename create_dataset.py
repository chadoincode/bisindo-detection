import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
# Inisialisasi deteksi maksimal 2 tangan
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        
        # Inisialisasi koordinat kosong (padding 0) untuk tangan kanan & kiri
        # 21 landmarks x 2 koordinat (x,y) = 42 slot per tangan
        right_hand_features = [0.0] * 42
        left_hand_features = [0.0] * 42

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Melakukan perulangan berdasarkan tangan yang terdeteksi
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Deteksi label tangan (Kanan atau Kiri) dari MediaPipe
                hand_label = results.multi_handedness[i].classification[0].label
                
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y

                temp_coords = []
                for landmark in hand_landmarks.landmark:
                    # Kurangi koordinat titik lain dengan titik acuan pergelangan tangan
                    temp_coords.append(landmark.x - base_x)
                    temp_coords.append(landmark.y - base_y)
                
                # Masukkan ke slot yang sesuai agar posisi fitur tidak tertukar
                if hand_label == 'Right':
                    right_hand_features = temp_coords
                elif hand_label == 'Left':
                    left_hand_features = temp_coords

            # Gabungkan fitur tangan kanan dan kiri (Total selalu 84 nilai)
            data_aux = right_hand_features + left_hand_features
            data.append(data_aux)
            labels.append(dir_)

# Simpan data hasil ekstraksi ke file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Ekstraksi dataset selesai! File data.pickle telah dibuat.")