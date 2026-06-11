import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

while True:
    ret, frame = cap.read()
    # Jangan gunakan cv2.flip jika ingin kameranya tidak terbalik (seperti di YouTube)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Mengambil label kanan/kiri dari MediaPipe
            hand_label = results.multi_handedness[i].classification[0].label
            
            # Ambil koordinat landmark pertama (pergelangan tangan) untuk posisi teks
            x = int(hand_landmarks.landmark[0].x * frame.shape[1])
            y = int(hand_landmarks.landmark[0].y * frame.shape[0])
            
            # Tampilkan tulisan "Right" atau "Left" di dekat tangan Anda di layar
            cv2.putText(frame, hand_label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.imshow('Cek Label Tangan', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()