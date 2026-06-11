import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

alphabet_labels = [chr(i) for i in range(ord('A'), ord('Z')+1)] 

print("=== TABEL HASIL PREDIKSI TIAP ALFABET ===")
report = classification_report(y_test, y_predict, target_names=alphabet_labels)
print(report)

# Opsional: Jika ingin menyimpan tabel tersebut ke dalam file Excel/CSV
report_dict = classification_report(y_test, y_predict, target_names=alphabet_labels, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv('laporan_prediksi_bisindo.csv', index=True)


cm = confusion_matrix(y_test, y_predict)

# Mengubah ke persen agar lebih mudah dibaca seperti di jurnal ilmiah (opsional)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(14, 10)) # Ukuran grafik diperbesar agar muat 26 huruf BISINDO
sns.heatmap(
    cm_percent, 
    annot=True,            # Menampilkan angka persentase di dalam kotak
    fmt='.1f',             # Format 1 angka di belakang koma (misal: 95.5%)
    cmap='Blues',          # Gradasi warna biru
    xticklabels=alphabet_labels, 
    yticklabels=alphabet_labels
)

plt.title('Grafik Akurasi & Prediksi Alfabet BISINDO (%)', fontsize=16, pad=20)
plt.xlabel('Label Prediksi (Hasil Sistem)', fontsize=12)
plt.ylabel('Label Sebenarnya (Ground Truth)', fontsize=12)
plt.tight_layout()

# Menyimpan grafik sebagai gambar untuk kebutuhan dokumen/jurnal
plt.savefig('grafik_akurasi_bisindo.png', dpi=300)
plt.show()