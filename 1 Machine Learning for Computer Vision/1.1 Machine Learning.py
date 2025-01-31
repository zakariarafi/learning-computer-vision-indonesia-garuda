"""
------------------------------------
Modul Computer Vision dengan Pendekatan Machine Learning (Bahasa Indonesia)
------------------------------------

Modul ini memparafrase materi tentang bagaimana visi komputer (computer vision) 
mencoba meniru cara kerja penglihatan manusia. Terdapat dua elemen utama dalam 
penglihatan: sistem sensor (mata) dan sistem kognitif (otak). 

Penglihatan manusia bekerja seperti ini:
1. Mata (sensor) menerima rangsangan cahaya, membentuk representasi visual.
2. Otak (sistem kognitif) memproses representasi tersebut dan memahami apa yang dilihat.

Dalam computer vision:
1. Bagian "image formation" berfokus pada perangkat keras (hardware) untuk menangkap gambar 
   atau video, misalnya kamera atau sensor lain.
2. Bagian "machine perception" menitikberatkan pada pemrosesan dan analisis gambar agar 
   komputer dapat "memahami" apa yang ada di dalam gambar tersebut.

Pendekatan modern yang meniru sistem kognitif manusia untuk computer vision banyak 
menggunakan metode machine learning (ML). Hal ini memungkinkan komputer mempelajari 
pola dalam gambar (atau dataset gambar) dan kemudian melakukan tugas seperti klasifikasi 
atau deteksi objek secara otomatis.

------------------------------------
1. Pengantar: Visi Komputer dan Contoh Sederhana
------------------------------------

Misalkan kita memiliki sebuah foto bunga daisy. Kita (manusia) dengan mudah dapat 
mengenalinya sebagai bunga daisy. Machine learning di sisi lain juga dapat diajarkan 
untuk mengenali foto daisy dengan cara:
- Menunjukkan banyak contoh foto daisy (beserta label atau jawabannya),
- Meminta model "belajar" pola bentuk, warna, dan karakteristik yang membedakan daisy 
  dari bunga lain,
- Saat diberikan foto baru yang belum pernah dilihat, model mencoba memprediksi apakah 
  itu daisy atau bukan.

------------------------------------
2. Dari Metode Tradisional Menuju Machine Learning
------------------------------------

Sebelum sekitar tahun 2010, metode pengolahan citra (image processing) untuk 
mengambil informasi dari gambar banyak menggunakan teknik seperti:
- Denoising (mengurangi noise pada gambar)
- Edge finding (menemukan tepi/batas objek)
- Texture detection (mendeteksi pola tekstur)
- Operasi morfologis (pengolahan bentuk)

Kemudian, AI (Artificial Intelligence) berkembang, khususnya sub-bidang Machine Learning, 
yang memungkinkan komputer "belajar" dari data. Ada pula pendekatan "expert systems" 
atau "sistem pakar" di mana aturan-aturan dibuat berdasarkan logika manusia ahli. 
Namun pendekatan tersebut kurang fleksibel dan memerlukan banyak aturan khusus 
(bespoke) yang sulit diperumum pada banyak kondisi.

Perkembangan deep learning di sekitar tahun 2012 dan seterusnya mengubah cara kerja 
computer vision. Klasifikasi gambar yang dulunya membutuhkan banyak filter dan logika 
pakar, kini dapat dilakukan lebih efektif hanya dengan memberikan data yang cukup 
dan memanfaatkan jaringan syaraf tiruan (neural network), terutama Convolutional 
Neural Network (CNN).

------------------------------------
3. Contoh Kasus: Daisy Versus Bunga Lain (Pendekatan ML vs Expert System)
------------------------------------

Pendekatan Machine Learning:
- Mengumpulkan dataset gambar bunga (daisy, mawar, tulip, dsb.) lengkap dengan label.
- Model CNN "belajar" ciri-ciri visual masing-masing bunga dari data.
- Saat mendapat gambar baru, model mencoba mengklasifikasikan sesuai kategori bunga yang paling cocok.

Pendekatan Expert System:
- Mewawancarai seorang ahli botani bagaimana membedakan bunga satu dengan yang lain.
- Membuat serangkaian filter untuk mendeteksi warna putih, kuning, hijau (misalnya untuk daisy), 
  mendeteksi bentuk daun, tepi bunga, dsb.
- Jika sesuai kriteria daisy, skor daisy naik. Jika sesuai kriteria tulip, skor tulip naik. 
  Begitu seterusnya. Kategori dengan skor tertinggi diambil sebagai prediksi.

Pendekatan berbasis aturan (expert system) ini sangat memakan waktu pembuatan (membutuhkan 
banyak penyesuaian manual) dan kurang fleksibel. Sementara dengan machine learning, 
kita cukup menyediakan data pelatihan (training data) yang memadai, melatih model, 
lalu model dapat mempelajari fitur-fitur yang relevan secara otomatis.

------------------------------------
4. Revolusi AlexNet (2012) dan Kebangkitan Deep Learning
------------------------------------

Tahun 2012, publikasi riset oleh Alex Krizhevsky, Ilya Sutskever, dan Geoffrey Hinton 
(AlexNet) menunjukkan peningkatan dramatis di ImageNet Large-Scale Visual 
Recognition Challenge (ILSVRC). Mereka mencapai error top-5 sekitar 15.3% saat itu, 
jauh lebih baik dibanding metode terbaik lain yang memiliki error di atas 26%. 

Kunci sukses AlexNet meliputi:
1. Penggunaan GPU (Graphics Processing Unit)
   - CNN (Convolutional Neural Network) membutuhkan komputasi besar. 
     Memanfaatkan GPU mempercepat proses training.
2. Penggunaan ReLU (Rectified Linear Unit) sebagai fungsi aktivasi
   - Fungsi aktivasi ini "non-saturating" sehingga mempercepat konvergensi 
     (pemodelan lebih cepat mencapai hasil yang baik).
3. Regularization
   - Karena ReLU bisa menyebabkan bobot jaringan tumbuh terlalu besar, 
     diperlukan teknik regularisasi (misal dropout) agar bobot model tetap stabil.
4. Model "lebih dalam" (deep)
   - Perpaduan ketiga hal di atas membuat mereka berani menambah lapisan (layer) 
     jaringan sehingga model lebih kompleks dan punya kemampuan ekstraksi fitur yang 
     lebih baik. Inilah yang disebut "Deep Learning".

Sejak keberhasilan AlexNet, CNN dan deep learning menjadi pendekatan utama dalam 
berbagai tugas computer vision (klasifikasi, deteksi objek, segmentasi, dsb.).

------------------------------------
5. Contoh Implementasi Python Sederhana dengan CNN (Keras)
------------------------------------

Di bawah ini adalah contoh kode sederhana menggunakan TensorFlow Keras untuk 
membangun model CNN. Contoh berikut hanya untuk ilustrasi dasar. 

CATATAN:
- Kode ini tidak memuat dataset bunga daisy sungguhan, namun hanya menunjukkan 
  struktur umum CNN. Anda dapat mengganti dataset dengan gambar bunga atau dataset lain 
  sesuai kebutuhan.
- Package yang diperlukan: tensorflow, keras, numpy (opsional), dsb.

"""

# --------------------------------------
# Contoh Implementasi CNN Sederhana
# --------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def simple_cnn_model(input_shape=(64, 64, 3), num_classes=5):
    """
    Membuat arsitektur CNN sederhana menggunakan Keras Sequential API.
    
    Argumen:
    - input_shape: Bentuk input gambar, misal 64x64 piksel dengan 3 channel warna (RGB).
    - num_classes: Jumlah kelas yang ingin diprediksi (misalnya 5 jenis bunga).
    
    Mengembalikan:
    - model: Objek model Keras yang siap dilatih.
    """
    model = models.Sequential()

    # Lapisan konvolusi pertama + pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Lapisan konvolusi kedua + pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Lapisan konvolusi ketiga + pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten sebelum masuk ke fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    
    # Output layer sesuai jumlah kelas
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Kompilasi model dengan optimizer dan loss function
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def example_training_procedure():
    """
    Prosedur sederhana untuk melatih model CNN di atas (hanya ilustrasi).
    
    - Kita buat data dummy untuk contoh (bukan data nyata).
    - Data X berukuran (100, 64, 64, 3), artinya 100 gambar ukuran 64x64 RGB.
    - Label y berukuran (100,), bernilai acak antara 0 hingga num_classes-1.
    """
    
    num_classes = 5
    model = simple_cnn_model(input_shape=(64, 64, 3), num_classes=num_classes)
    
    # Contoh data dummy
    X_dummy = np.random.random((100, 64, 64, 3))  # 100 gambar acak
    y_dummy = np.random.randint(0, num_classes, 100)  # 100 label acak

    # Melatih model (hanya beberapa epoch, contoh)
    model.fit(X_dummy, y_dummy, epochs=5, batch_size=10)
    
    # Evaluasi model pada data dummy yang sama (hanya untuk contoh)
    loss, accuracy = model.evaluate(X_dummy, y_dummy)
    
    print("Loss (Dummy):", loss)
    print("Accuracy (Dummy):", accuracy)

if __name__ == "__main__":
    """
    Jika dijalankan sebagai skrip utama, modul akan mengeksekusi contoh prosedur 
    pelatihan di atas. Di dunia nyata, Anda akan mengganti X_dummy dan y_dummy 
    dengan dataset sebenarnya (mis. gambar bunga daisy, mawar, dll.).
    """
    example_training_procedure()
