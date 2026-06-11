import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model yang sudah dilatih (pastikan kamu sudah menjalankan script training bawaan video)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

labels_dict = {
    0: 'A', 
    1: 'B', 
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
    } 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    right_hand_features = [0.0] * 42
    left_hand_features = [0.0] * 42
    
    # List koordinat untuk keperluan membuat bounding box gambar
    x_coords_for_box = []
    y_coords_for_box = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Gambar titik landmark di layar untuk visualisasi
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            hand_label = results.multi_handedness[i].classification[0].label
            
            temp_coords = []
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            temp_coords = []
            for landmark in hand_landmarks.landmark:
                # Kurangi koordinat titik lain dengan titik acuan pergelangan tangan
                temp_coords.append(landmark.x - base_x)
                temp_coords.append(landmark.y - base_y)
                
                # Masukkan koordinat pixel asli untuk kalkulasi bounding box nanti
                x_coords_for_box.append(int(landmark.x * W))
                y_coords_for_box.append(int(landmark.y * H))

            if hand_label == 'Right':
                right_hand_features = temp_coords
            elif hand_label == 'Left':
                left_hand_features = temp_coords

        # Gabungkan fitur secara konsisten seperti saat pembuatan dataset
        data_aux = right_hand_features + left_hand_features

        # Prediksi menggunakan model Random Forest
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Membuat satu Bounding Box yang otomatis membungkus kedua tangan di layar
        x1, y1 = min(x_coords_for_box) - 20, min(y_coords_for_box) - 20
        x2, y2 = max(x_coords_for_box) + 20, max(y_coords_for_box) + 20

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('Deteksi BISINDO Real-time', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()