# Drowsy Driver Detection System using Vision Transformer (ViT)

Sistem deteksi kantuk pengemudi secara *real-time* yang memanfaatkan teknologi **Computer Vision** dan **Deep Learning**. Proyek ini menggunakan arsitektur **Vision Transformer (ViT)** untuk mengklasifikasikan status pengemudi menjadi "Drowsy" (mengantuk) atau "Not Drowsy" (terjaga) melalui input kamera.

## Informasi Model

Model ini adalah versi *fine-tuned* dari `google/vit-base-patch16-224-in21k` yang dilatih khusus menggunakan dataset **UTA-RLDD (Real-Life Drowsiness Dataset)**.

* **Repositori Model:** [chbh7051/driver-drowsiness-detection](https://huggingface.co/chbh7051/driver-drowsiness-detection)
* **Akurasi:** 97.52%
* **Loss:** 0.0800
* **Framework:** Transformers 4.27.4 & PyTorch

### Hiperparameter Pelatihan

* **Learning Rate:** 0.0002
* **Batch Size:** 16 (Train) / 8 (Eval)
* **Optimizer:** Adam dengan betas=(0.9, 0.999)
* **Epochs:** 2.0

---

## Catatan Penting untuk Pengujian (Demo)

**Model ini dioptimalkan khusus untuk kondisi pengemudi yang berada di dalam kendaraan.**

Perlu diperhatikan bahwa akurasi deteksi sangat bergantung pada **konteks lingkungan berkendara**:

* **Kondisi Ideal:** Model akan bekerja sangat akurat (97.52%) ketika subjek benar-benar berada di kursi pengemudi dengan sudut pandang kamera yang sesuai.
* **Keterbatasan Demo:** Nilai prediksi mungkin menjadi tidak akurat jika diuji dalam posisi duduk biasa (misal: di depan meja kerja) dan mencoba berpura-pura mengantuk. Hal ini dikarenakan model dilatih menggunakan dataset *Real-Life Drowsiness* yang menangkap gestur, pencahayaan, dan latar belakang spesifik di dalam kabin kendaraan.

---

## Fitur Utama

* **Face Detection:** Menggunakan Haar Cascade untuk melokalisasi wajah pengemudi secara *real-time*.
* **ViT Classifier:** Menggunakan Vision Transformer (ViT-Base) untuk klasifikasi status kantuk yang lebih mendalam dibandingkan CNN tradisional.
* **Dual Alert System:** * **Visual Alert:** Overlay merah pada layar dengan teks peringatan saat ambang batas kantuk tercapai.
* **Audio Alert:** Peringatan suara sirine menggunakan `alert.wav` melalui library Pygame.


* **Real-time Analytics:** Menampilkan FPS, total frame, dan statistik deteksi langsung pada antarmuka video.
* **Automated Logging:** Menyimpan riwayat deteksi ke dalam file `data/drowsy_log.csv` untuk keperluan analisis.

## Struktur Proyek

* `drowsy_detection.py`: Skrip utama yang menggabungkan semua komponen deteksi.
* `utils/face_detector.py`: Logika deteksi wajah menggunakan OpenCV Haar Cascade.
* `utils/preprocessor.py`: Pemrosesan gambar (RGB conversion & tensor format) untuk input ViT.
* `utils/alert_system.py`: Pengatur peringatan suara dan visual.
* `models/`: Folder penyimpanan bobot model (`pytorch_model.bin`) dan konfigurasi.
* `assets/`: Aset pendukung seperti file suara `alert.wav`.

## Instalasi & Persiapan

1. **Clone Repositori:**
```bash
git clone https://github.com/username/drowsy-driver-detection.git
cd drowsy-driver-detection

```


2. **Instal Dependensi:**
Pastikan Python telah terinstal, lalu jalankan:
```bash
pip install -r requirements.txt

```


3. **Persiapan Model:**
Pastikan file `pytorch_model.bin` berada di folder `models/`. Jika menggunakan Git LFS, gunakan perintah `git lfs pull`.

## Cara Penggunaan

Jalankan program utama:

```bash
python drowsy_detection.py

```

**Kontrol:**

* **'q'**: Berhenti dan keluar dari aplikasi.
* **'r'**: Reset statistik deteksi (total frames, alert count).

## Mekanisme Peringatan

Untuk menghindari *false alarm* akibat kedipan mata normal, sistem menggunakan logika hitung mundur:

1. Status "drowsy" harus terdeteksi selama **15 frame berturut-turut** (~0.5 detik pada 30 FPS).
2. Setelah ambang batas (`drowsy_threshold`) terlampaui, alarm visual dan suara akan aktif secara simultan.
3. Alarm akan otomatis berhenti jika pengemudi kembali terdeteksi dalam status "awake".