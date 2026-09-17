# Analisis Supervised dan Unsupervised Learning

## 📌 Overview

Repository ini berisi hasil pembelajaran dan eksperimen **Machine Learning menggunakan Python dan Google Colab** berdasarkan materi *Introduction to Machine Learning with Python*, khususnya Chapter 2 tentang **Supervised Learning** dan Chapter 3 tentang **Unsupervised Learning and Preprocessing**.

Eksperimen dilakukan terhadap 13 algoritma yang terbagi menjadi:

* **7 Supervised Learning**
* **6 Unsupervised Learning**

Tujuan dari praktikum ini bukan hanya untuk mengetahui cara menjalankan algoritma menggunakan `scikit-learn`, tetapi juga untuk memahami **perbedaan cara kerja model, pengaruh parameter, karakteristik data yang sesuai, serta bagaimana menginterpretasikan hasil eksperimen**.

Buku menempatkan KNN, Linear Models, Naive Bayes, Decision Trees, Ensembles of Decision Trees, Kernelized SVM, dan Neural Networks sebagai algoritma supervised. Sementara PCA, NMF, t-SNE, k-Means, Agglomerative Clustering, dan DBSCAN dibahas dalam bagian unsupervised learning.

---

# 🧠 1. Analisis Supervised Learning

## 1.1 Konsep Dasar

Pada supervised learning, model belajar dari data yang mempunyai **fitur dan target**. Artinya, selama proses training model memiliki informasi mengenai jawaban yang benar.

Dari eksperimen yang dilakukan, pola umum supervised learning yang saya temukan adalah:

```text
Dataset
   ↓
Feature + Target
   ↓
Training Data
   ↓
Model belajar pola
   ↓
Testing Data
   ↓
Prediksi
   ↓
Evaluasi
```

Hal yang menurut saya paling penting dari proses ini adalah **model tidak cukup hanya memiliki nilai training yang tinggi**. Model juga harus mampu bekerja pada data yang belum pernah dilihat.

Hal tersebut berkaitan dengan konsep **generalization**. Jika model terlalu menyesuaikan diri terhadap training data, maka model dapat mengalami **overfitting**. Sebaliknya, jika model terlalu sederhana dan tidak mampu menangkap pola data, model dapat mengalami **underfitting**.

---

# 2. Analisis Setiap Model Supervised

## 2.1 K-Nearest Neighbors

KNN memiliki pendekatan yang cukup intuitif. Model tidak membangun aturan klasifikasi yang kompleks, tetapi melihat **tetangga terdekat** dari data yang akan diprediksi.

Dari eksperimen KNN, saya memahami bahwa parameter `n_neighbors` menjadi bagian yang penting. Ketika jumlah tetangga kecil, model cenderung mengikuti pola lokal data. Akibatnya, model dapat menjadi lebih sensitif terhadap perubahan atau noise.

Sebaliknya, ketika jumlah tetangga diperbesar, keputusan model menjadi lebih umum karena mempertimbangkan lebih banyak data.

Hal ini menunjukkan adanya hubungan antara **kompleksitas model dan generalisasi**. KNN dengan nilai K yang kecil dapat menjadi sangat fleksibel, sedangkan nilai K yang lebih besar membuat model lebih sederhana.

Menurut materi buku, KNN juga cocok digunakan sebagai baseline terutama pada dataset yang relatif kecil karena konsepnya mudah dijelaskan.

**Pemahaman dari eksperimen:** KNN mudah digunakan dan mudah dipahami, tetapi pemilihan K sangat menentukan perilaku model.

---

## 2.2 Linear Models

Linear Models memiliki pendekatan yang berbeda dari KNN. Model mencoba menemukan hubungan matematis yang relatif sederhana antara fitur dan target.

Dalam eksperimen saya menggunakan **Linear Regression** dan **Ridge Regression**. Dari sini terlihat bahwa model linear dapat digunakan sebagai model awal karena struktur modelnya relatif sederhana.

Eksperimen Ridge memberikan pemahaman tambahan mengenai **regularisasi**. Parameter `alpha` digunakan untuk mengontrol seberapa kuat regularisasi diberikan.

Analisis yang saya dapatkan adalah semakin kompleks permasalahan, model linear belum tentu mampu mengikuti seluruh pola data. Namun, kesederhanaan model menjadi keuntungan ketika data berukuran besar atau memiliki banyak fitur.

Buku juga menyebutkan linear models sebagai salah satu model pertama yang baik untuk dicoba pada dataset baru.

**Pemahaman dari eksperimen:** model sederhana bukan berarti tidak berguna. Linear Models dapat menjadi baseline untuk mengetahui apakah masalah yang dihadapi sudah dapat diselesaikan dengan hubungan yang relatif sederhana.

---

## 2.3 Naive Bayes

Naive Bayes menggunakan pendekatan probabilistik untuk melakukan klasifikasi.

Hal yang menarik dari eksperimen Naive Bayes adalah model ini tidak membutuhkan proses yang serumit beberapa model lain, tetapi tetap dapat menghasilkan klasifikasi dengan cepat.

Saya juga mempelajari bahwa terdapat beberapa varian Naive Bayes, seperti **GaussianNB, BernoulliNB, dan MultinomialNB**. Masing-masing memiliki karakteristik data yang berbeda.

Dari sini saya memahami bahwa memilih algoritma bukan hanya berdasarkan nama model, tetapi juga berdasarkan **bentuk data yang digunakan**.

Buku menjelaskan bahwa Naive Bayes sangat cepat dalam training maupun prediksi dan dapat menjadi baseline yang baik, khususnya pada dataset besar dan berdimensi tinggi.

**Pemahaman dari eksperimen:** kecepatan dan kesederhanaan dapat menjadi pertimbangan penting ketika dataset berukuran besar.

---

## 2.4 Decision Trees

Decision Tree menurut saya merupakan salah satu model yang paling mudah dipahami secara visual.

Model membangun serangkaian pertanyaan atau kondisi **if/else** hingga menghasilkan keputusan. Pada eksperimen dengan dataset two-moons, saya dapat melihat bagaimana perubahan struktur tree memengaruhi pemisahan kelas.

Parameter seperti `max_depth` menjadi penting karena menentukan seberapa dalam tree dapat berkembang.

Jika tree terlalu dalam, model dapat menjadi sangat kompleks dan mengikuti training data secara berlebihan. Jika terlalu dangkal, model mungkin belum cukup mampu menangkap pola data.

Buku menjelaskan bahwa Decision Tree mempelajari hierarki pertanyaan if/else dari data dan dapat digunakan untuk classification maupun regression.

**Pemahaman dari eksperimen:** Decision Tree mudah dijelaskan kepada manusia, tetapi kedalaman tree perlu dikontrol agar kompleksitas model tetap sesuai.

---

## 2.5 Random Forest

Random Forest merupakan pengembangan dari konsep Decision Tree dengan menggunakan banyak tree.

Dari eksperimen, saya memahami alasan mengapa ensemble dapat lebih stabil dibandingkan satu Decision Tree. Setiap tree dapat menghasilkan keputusan yang berbeda, kemudian hasilnya digabungkan.

Dengan pendekatan tersebut, model tidak terlalu bergantung pada satu struktur tree.

Namun, konsekuensinya adalah interpretasi model menjadi lebih sulit. Jika Decision Tree tunggal dapat divisualisasikan secara langsung, Random Forest terdiri dari banyak tree sehingga proses pengambilan keputusannya tidak sesederhana satu tree.

Buku menjelaskan Random Forest sebagai model yang robust dan powerful serta tidak membutuhkan scaling data.

**Pemahaman dari eksperimen:** ensemble menunjukkan bahwa menggabungkan beberapa model yang relatif sederhana dapat menghasilkan sistem yang lebih kuat.

---

## 2.6 Kernelized SVM

Kernelized SVM memberikan pemahaman yang berbeda mengenai bagaimana model dapat menangani pola data yang tidak linear.

Pada eksperimen, saya menggunakan **RBF kernel** dan mengamati pengaruh `C` serta `gamma`.

Parameter tersebut memengaruhi kompleksitas decision boundary. Perubahan parameter dapat membuat batas klasifikasi menjadi lebih fleksibel atau lebih sederhana.

Konsep **support vectors** juga menjadi bagian penting. Data tertentu memiliki peran besar dalam menentukan posisi decision boundary.

Dari eksperimen tersebut saya memahami bahwa SVM dapat menghasilkan batas klasifikasi yang kompleks, tetapi penggunaannya membutuhkan perhatian lebih terhadap parameter dan scaling data.

Buku juga menyebutkan bahwa SVM powerful untuk dataset berukuran sedang, tetapi sensitif terhadap parameter dan scaling.

**Pemahaman dari eksperimen:** SVM menunjukkan bahwa performa model tidak hanya ditentukan oleh algoritmanya, tetapi juga oleh konfigurasi parameter dan representasi data.

---

## 2.7 Neural Networks / Deep Learning

Neural Network menjadi model dengan konsep paling kompleks di antara supervised model yang dicoba.

Dalam eksperimen menggunakan `MLPClassifier`, saya mencoba beberapa konfigurasi hidden layer, jumlah neuron, fungsi aktivasi, dan nilai `alpha`.

Perubahan konfigurasi tersebut dapat mengubah bentuk decision boundary. Artinya, arsitektur jaringan memiliki hubungan langsung dengan kemampuan model dalam mempelajari pola.

Namun, model yang lebih kompleks juga membutuhkan perhatian lebih terhadap parameter. Jika konfigurasi tidak sesuai, model tidak otomatis menghasilkan hasil yang lebih baik.

Buku menjelaskan bahwa neural networks dapat membangun model yang sangat kompleks, tetapi sensitif terhadap scaling dan pemilihan parameter.

**Pemahaman dari eksperimen:** Neural Network memberikan fleksibilitas tinggi, tetapi fleksibilitas tersebut juga meningkatkan kebutuhan terhadap tuning dan pemahaman model.

---

# 🔎 3. Analisis Antar-Model Supervised

Jika dibandingkan, ketujuh model supervised yang dicoba menunjukkan bahwa terdapat **trade-off antara kesederhanaan, fleksibilitas, interpretasi, dan kebutuhan komputasi**.

KNN relatif mudah dipahami, Linear Models sederhana, Naive Bayes cepat, Decision Tree mudah divisualisasikan, Random Forest lebih robust, SVM mampu membuat decision boundary kompleks, sedangkan Neural Network memiliki fleksibilitas yang sangat tinggi.

Hal ini membuat saya memahami bahwa memilih model tidak seharusnya hanya berdasarkan model yang terlihat paling kompleks.

Secara konseptual, proses pemilihan model dapat dilakukan seperti:

```text
Pahami dataset
      ↓
Tentukan masalah
      ↓
Coba model sederhana
      ↓
Evaluasi hasil
      ↓
Analisis kekurangan
      ↓
Coba model yang lebih kompleks
      ↓
Bandingkan hasil
```

Pendekatan ini juga sesuai dengan pembahasan buku yang menyarankan untuk memulai dari model sederhana sebelum berpindah ke model yang lebih kompleks.

---

# 🧩 4. Analisis Unsupervised Learning

## 4.1 Konsep Dasar

Unsupervised learning memiliki kondisi yang berbeda karena tidak menggunakan target sebagai jawaban yang harus diprediksi.

Dari eksperimen yang dilakukan, saya melihat bahwa unsupervised learning dapat dibagi menjadi dua tujuan besar:

### Dimensionality Reduction / Representation

* PCA
* NMF
* t-SNE

### Clustering

* k-Means
* Agglomerative Clustering
* DBSCAN

Karena tidak terdapat label sebagai jawaban benar, analisis hasilnya juga berbeda dengan supervised learning.

---

# 5. Analisis Setiap Model Unsupervised

## 5.1 PCA

PCA digunakan untuk mengurangi dimensi data dengan mencari representasi baru yang lebih ringkas.

Dari praktik PCA, saya memahami bahwa jumlah fitur yang banyak tidak selalu berarti informasi yang diperoleh lebih mudah dipahami.

PCA membantu menyederhanakan representasi data sehingga hubungan atau pola tertentu dapat lebih mudah diamati.

Namun, hasil PCA berupa komponen baru sehingga interpretasi setiap komponen tidak selalu langsung sama dengan fitur asli.

**Pemahaman dari eksperimen:** reduksi dimensi dapat membantu mengatasi kompleksitas data tanpa harus langsung membuang seluruh informasi dari dataset.

---

## 5.2 NMF

NMF memiliki tujuan yang mirip dengan PCA dalam hal menemukan representasi data yang lebih sederhana, tetapi menggunakan pendekatan yang berbeda.

Hal penting yang saya pahami adalah NMF menghasilkan komponen non-negatif. Karakteristik tersebut membuat NMF menarik ketika representasi berupa bagian-bagian atau kontribusi positif lebih mudah diinterpretasikan.

Dengan demikian, PCA dan NMF sama-sama dapat digunakan untuk mendapatkan representasi baru, tetapi konsep matematis dan karakteristik hasilnya berbeda.

**Pemahaman dari eksperimen:** dua algoritma yang sama-sama melakukan reduksi atau ekstraksi representasi belum tentu memberikan representasi yang sama.

---

## 5.3 t-SNE

t-SNE merupakan algoritma yang lebih diarahkan untuk **visualisasi data berdimensi tinggi**.

Pada eksperimen menggunakan digits dataset, data yang memiliki banyak fitur direpresentasikan menjadi dua dimensi.

Hasil tersebut membantu melihat apakah data tertentu membentuk kelompok yang secara visual terlihat berdekatan.

Namun, saya memahami bahwa t-SNE tidak sama dengan clustering. Jika terlihat beberapa kelompok pada visualisasi t-SNE, hal tersebut tidak otomatis berarti algoritma telah memberikan label cluster.

Buku juga menjelaskan bahwa t-SNE terutama digunakan untuk visualisasi dan tidak menyediakan metode `transform` untuk menerapkan representasi yang sama pada data baru.

**Pemahaman dari eksperimen:** visualisasi dapat membantu memahami struktur data, tetapi visualisasi tidak boleh langsung dianggap sebagai hasil klasifikasi.

---

# 6. Analisis Clustering

## 6.1 k-Means

k-Means mengelompokkan data berdasarkan kedekatan terhadap centroid.

Dari praktiknya, saya memahami bahwa pengguna harus menentukan jumlah cluster terlebih dahulu. Hal ini menjadi salah satu perbedaan penting dibandingkan algoritma seperti DBSCAN.

Kelebihan pendekatan ini adalah konsepnya sederhana dan hasilnya relatif mudah divisualisasikan.

Namun, ketika jumlah cluster yang ditentukan tidak sesuai dengan struktur sebenarnya, hasil pengelompokan juga dapat menjadi kurang representatif.

**Pemahaman dari eksperimen:** k-Means cocok ketika kita mempunyai alasan atau analisis tertentu mengenai jumlah kelompok yang ingin dicari.

---

## 6.2 Agglomerative Clustering

Agglomerative Clustering menggunakan pendekatan hierarchical clustering.

Berbeda dari k-Means yang menggunakan centroid sebagai pusat cluster, Agglomerative Clustering membangun kelompok secara bertahap berdasarkan hubungan antar data.

Dari eksperimen, saya memahami bahwa algoritma ini memberikan sudut pandang yang berbeda terhadap struktur kelompok. Data tidak hanya dipandang sebagai kumpulan titik terhadap pusat cluster, tetapi dapat dipahami sebagai struktur hierarki.

Buku membahas Agglomerative Clustering sebagai hierarchical clustering dan juga membahas pilihan linkage yang memengaruhi proses penggabungan cluster.

**Pemahaman dari eksperimen:** cara mendefinisikan hubungan antar data dapat menghasilkan struktur clustering yang berbeda.

---

## 6.3 DBSCAN

DBSCAN menggunakan konsep **kepadatan data**.

Dari eksperimen, saya memahami bahwa DBSCAN memiliki karakteristik yang berbeda dari k-Means karena dapat mengidentifikasi data yang berada di area dengan kepadatan rendah sebagai noise.

Hal tersebut berguna ketika dataset memiliki bentuk cluster yang tidak sederhana.

Namun, hasil DBSCAN sangat dipengaruhi oleh parameter seperti `eps` dan `min_samples`. Oleh karena itu, parameter tersebut perlu disesuaikan dengan karakteristik dataset.

**Pemahaman dari eksperimen:** clustering tidak selalu harus berdasarkan centroid. Kepadatan data juga dapat digunakan untuk menentukan struktur kelompok.

---

# ⚖️ 7. Perbandingan Unsupervised Learning

| Algoritma     | Pendekatan                | Hal yang Dipelajari                            |
| ------------- | ------------------------- | ---------------------------------------------- |
| PCA           | Reduksi dimensi           | Mencari representasi data yang lebih sederhana |
| NMF           | Representasi fitur        | Membentuk komponen non-negatif                 |
| t-SNE         | Manifold learning         | Memvisualisasikan struktur lokal data          |
| k-Means       | Centroid-based clustering | Mengelompokkan berdasarkan kedekatan centroid  |
| Agglomerative | Hierarchical clustering   | Membangun cluster secara bertahap              |
| DBSCAN        | Density-based clustering  | Mencari kelompok berdasarkan kepadatan         |

Buku sendiri mengelompokkan k-Means, Agglomerative Clustering, dan DBSCAN sebagai metode clustering, sedangkan t-SNE berada pada bagian manifold learning dan PCA/NMF pada dimensionality reduction serta feature extraction.

---

# 🔬 8. Analisis Perbandingan Supervised vs Unsupervised

Perbedaan paling mendasar yang saya pahami setelah melakukan praktik adalah **tujuan pembelajarannya**.

Pada supervised learning, saya sudah mengetahui target yang ingin diprediksi. Karena itu, saya dapat membandingkan hasil prediksi dengan target sebenarnya.

Pada unsupervised learning, saya tidak memiliki jawaban yang harus diprediksi. Saya justru mencoba menemukan struktur yang terdapat dalam data.

Contohnya:

```text
SUPERVISED
Data → Model → Prediksi → Bandingkan dengan Label

UNSUPERVISED
Data → Model → Struktur/Pola → Analisis
```

Hal ini menyebabkan proses evaluasinya juga berbeda.

Pada supervised learning, score seperti accuracy atau R² dapat digunakan tergantung jenis permasalahannya. Sedangkan pada unsupervised learning, hasil perlu dilihat dari struktur cluster, visualisasi, atau metrik clustering tertentu.

---

# 📈 9. Analisis Kompleksitas Model

Dari seluruh eksperimen, saya melihat adanya pola bahwa **semakin fleksibel sebuah model, semakin besar pula kebutuhan untuk mengatur kompleksitasnya**.

Contohnya:

* KNN → `n_neighbors`
* Linear/Ridge → `alpha`
* Decision Tree → `max_depth`
* SVM → `C` dan `gamma`
* Neural Network → hidden layers, jumlah neuron, activation, `alpha`
* k-Means → jumlah cluster
* DBSCAN → `eps` dan `min_samples`

Parameter tersebut bukan sekadar angka tambahan dalam kode. Parameter menentukan **bagaimana model memandang pola data**.

Karena itu, salah satu pelajaran penting dari praktikum ini adalah bahwa menjalankan:

```python
model.fit(X_train, y_train)
```

belum berarti proses Machine Learning sudah selesai.

Masih diperlukan:

```text
Training
   ↓
Evaluasi
   ↓
Analisis hasil
   ↓
Tuning parameter
   ↓
Evaluasi kembali
```

---

# 🧠 10. Hal yang Paling Saya Pahami dari Praktikum

Setelah mencoba 13 algoritma, saya mendapatkan beberapa pemahaman yang lebih jelas dibandingkan hanya mempelajari teori.

### 1. Tidak ada satu algoritma yang cocok untuk semua masalah

Setiap algoritma memiliki asumsi dan karakteristik yang berbeda. Karena itu, pemilihan algoritma harus melihat bentuk data dan tujuan permasalahan.

### 2. Parameter dapat mengubah perilaku model

Percobaan terhadap `max_depth`, `alpha`, `C`, `gamma`, hidden layer, jumlah cluster, dan parameter lainnya menunjukkan bahwa parameter memiliki pengaruh terhadap kompleksitas model.

### 3. Model sederhana tetap penting

Model sederhana dapat digunakan sebagai **baseline**. Jika model sederhana sudah memberikan hasil yang baik, kita memiliki dasar untuk membandingkan apakah model yang lebih kompleks benar-benar memberikan peningkatan.

### 4. Visualisasi membantu memahami Machine Learning

Decision boundary, hasil PCA, visualisasi t-SNE, dan hasil clustering membuat saya lebih mudah memahami bagaimana model melihat data.

### 5. Menjalankan kode berbeda dengan memahami model

Hal paling penting dari praktikum ini adalah saya mulai memahami bahwa Machine Learning bukan sekadar menggunakan library `scikit-learn`.

Saya perlu mengetahui:

* data apa yang digunakan,
* apakah data memiliki label,
* tujuan model,
* parameter yang digunakan,
* hasil yang diperoleh,
* kemungkinan overfitting atau underfitting,
* serta alasan mengapa model tersebut digunakan.

---

# 🎯 11. Kesimpulan

Eksperimen terhadap 13 algoritma memberikan gambaran bahwa Machine Learning memiliki berbagai pendekatan untuk menyelesaikan permasalahan yang berbeda.

**Supervised Learning** lebih terarah karena model belajar menggunakan target. Dari tujuh model yang dicoba, saya melihat perbedaan pendekatan mulai dari model sederhana seperti Linear Models dan KNN sampai model yang lebih kompleks seperti Random Forest, SVM, dan Neural Networks.

**Unsupervised Learning** memiliki tantangan yang berbeda karena tidak terdapat target sebagai jawaban. PCA, NMF, dan t-SNE membantu memahami representasi atau struktur data, sedangkan k-Means, Agglomerative Clustering, dan DBSCAN mencoba menemukan kelompok dalam data dengan pendekatan yang berbeda.

Secara keseluruhan, hasil praktikum membuat saya memahami bahwa **Machine Learning bukan tentang mencari algoritma yang paling kompleks, tetapi tentang memilih pendekatan yang sesuai dengan data dan tujuan permasalahan**.

Sebagai mahasiswa semester 5 yang sedang mempelajari Machine Learning, pemahaman ini menjadi dasar penting sebelum masuk ke tahap yang lebih lanjut seperti **feature engineering, model evaluation, hyperparameter tuning, pipeline, dan penerapan model pada dataset nyata**.

---

# 📚 Referensi

Müller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.

Materi eksperimen menggunakan Python, Google Colab, dan `scikit-learn`.
