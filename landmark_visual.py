import os
import cv2
import mediapipe as mp

# 1. Inisialisasi MediaPipe Hands dan Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gunakan konfigurasi 2 tangan seperti pada proyek BISINDO Anda
hands = mp_hands.Hands(
    static_image_mode=True,  # Diaktifkan TRUE karena memproses gambar statis
    max_num_hands=2,         # Deteksi maksimal dua tangan untuk BISINDO
    min_detection_confidence=0.5
)

# 2. Tentukan lokasi gambar yang ingin diambil (Ubah sesuai folder Anda)
# Contoh: Mengambil gambar pertama (0.jpg) dari folder alfabet 'A'
PATH_GAMBAR = './data/11/0.jpg'  # Sesuaikan dengan nama folder dataset Anda

if not os.path.exists(PATH_GAMBAR):
    print(f"Error: Gambar tidak ditemukan di {PATH_GAMBAR}. Pastikan path sudah benar!")
else:
    # 3. Baca gambar menggunakan OpenCV
    image = cv2.imread(PATH_GAMBAR)
    
    # Salin gambar asli untuk digambar landmark-nya
    annotated_image = image.copy()
    
    # Konversi warna dari BGR ke RGB (karena MediaPipe memproses format RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Proses gambar untuk mendeteksi landmark tangan
    results = hands.process(image_rgb)

    # 4. Jika tangan terdeteksi, gambar skeletal landmark-nya
    if results.multi_hand_landmarks:
        print(f"Berhasil mendeteksi {len(results.multi_hand_landmarks)} tangan!")
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar titik koordinat dan garis penghubung bawaan MediaPipe
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
        # 5. Tampilkan hasilnya di layar komputer Anda
        cv2.imshow('Visualisasi Landmark untuk Jurnal', annotated_image)
        print("Jendela gambar terbuka. Tekan tombol APA SAJA di keyboard untuk menutup.")
        
        # 6. Simpan gambar hasil visualisasi ke folder lokal untuk lampiran
        nama_output = 'lampiran_landmark_bisindo.jpg'
        cv2.imwrite(nama_output, annotated_image)
        print(f"Gambar lampiran berhasil disimpan dengan nama: {nama_output}")
        
        cv2.waitKey(0)  # Menunggu input tombol untuk menutup jendela
        cv2.destroyAllWindows()
    else:
        print("MediaPipe gagal mendeteksi tangan pada gambar tersebut. Coba gunakan gambar sampel lain.")

hands.close()