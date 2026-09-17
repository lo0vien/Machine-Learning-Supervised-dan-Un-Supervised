# Analisis Supervised dan Unsupervised Learning

## 📌 Deskripsi

Repository ini berisi hasil pembelajaran dan implementasi beberapa algoritma **Machine Learning** menggunakan Python dan Google Colab. Materi yang dipelajari mengacu pada buku *Introduction to Machine Learning with Python*, khususnya **Chapter 2: Supervised Learning** dan **Chapter 3: Unsupervised Learning and Preprocessing**.

Pada praktiknya, saya mencoba menjalankan berbagai algoritma menggunakan dataset dan contoh yang digunakan dalam pembelajaran. Dari eksperimen tersebut, saya dapat melihat bagaimana setiap algoritma bekerja, bagaimana hasil prediksi atau pengelompokan terbentuk, serta bagaimana karakteristik setiap model berbeda satu sama lain.

Secara keseluruhan terdapat **13 model/algoritma** yang dipelajari, terdiri dari **7 algoritma Supervised Learning** dan **6 algoritma Unsupervised Learning**.

---

# 🧠 1. Supervised Learning

**Supervised Learning** merupakan pendekatan Machine Learning ketika model belajar menggunakan data yang memiliki **target atau label**. Model belajar hubungan antara fitur input dan target sehingga dapat digunakan untuk melakukan prediksi terhadap data baru.

Dalam praktik yang dilakukan, sebagian besar algoritma supervised diuji menggunakan proses **training dan testing**. Hal ini membantu melihat apakah model tidak hanya mampu mempelajari data training, tetapi juga dapat melakukan generalisasi terhadap data yang belum pernah dilihat sebelumnya.

Buku juga menjelaskan pentingnya generalisasi, overfitting, dan underfitting. Model yang terlalu fokus terhadap data training dapat mengalami overfitting sehingga performanya terhadap data baru menjadi kurang baik.

### Model yang Dipelajari

| No. | Model               | Fokus Pembelajaran                           |
| --- | ------------------- | -------------------------------------------- |
| 1   | K-Nearest Neighbors | Klasifikasi berdasarkan kedekatan data       |
| 2   | Linear Models       | Hubungan linear untuk klasifikasi/regresi    |
| 3   | Naive Bayes         | Klasifikasi berbasis probabilitas            |
| 4   | Decision Trees      | Pengambilan keputusan berbentuk if/else      |
| 5   | Random Forest       | Gabungan beberapa decision tree              |
| 6   | Kernelized SVM      | Pemisahan kelas menggunakan kernel           |
| 7   | Neural Networks     | Pembelajaran pola menggunakan jaringan saraf |

---

## 1.1 K-Nearest Neighbors (KNN)

KNN merupakan algoritma yang melakukan prediksi berdasarkan beberapa data yang memiliki jarak paling dekat dengan data yang akan diprediksi.

Dari percobaan di Google Colab, saya memahami bahwa **nilai K sangat memengaruhi hasil model**. Nilai K yang kecil membuat model lebih sensitif terhadap data di sekitarnya, sedangkan nilai K yang lebih besar membuat keputusan model menjadi lebih umum.

KNN cukup mudah dipahami karena konsep dasarnya hanya mencari tetangga terdekat. Namun, model ini dapat menjadi kurang praktis ketika jumlah data semakin besar karena proses prediksi perlu mempertimbangkan jarak terhadap data training.

---

## 1.2 Linear Models

Linear Models mencoba menemukan hubungan linear antara fitur dan target. Pada praktiknya saya mencoba **Linear Regression** serta **Ridge Regression**.

Dari eksperimen tersebut, saya dapat melihat nilai **training score dan test score** untuk mengetahui kemampuan model. Ridge juga memperkenalkan parameter `alpha` yang digunakan untuk mengatur regularisasi.

Hal yang saya pahami adalah bahwa model linear relatif sederhana dan cocok digunakan sebagai salah satu model awal ketika menghadapi dataset baru. Buku juga menyebutkan linear models sebagai salah satu algoritma yang baik untuk dicoba terlebih dahulu, terutama pada dataset besar atau berdimensi tinggi.

---

## 1.3 Naive Bayes

Naive Bayes merupakan algoritma klasifikasi yang menggunakan pendekatan probabilitas. Pada praktiknya saya mencoba **Gaussian Naive Bayes** dan melihat konsep **Bernoulli Naive Bayes** melalui data biner.

Hal yang menarik dari Naive Bayes adalah proses pelatihannya relatif cepat dan konsepnya cukup sederhana. Model ini juga dapat digunakan pada data berdimensi tinggi.

Dari pembelajaran, saya memahami bahwa Naive Bayes lebih berfokus pada perhitungan probabilitas dibandingkan mencari batas keputusan yang kompleks. Buku menjelaskan bahwa Naive Bayes sangat cepat dan dapat menjadi baseline yang baik, khususnya untuk dataset berukuran besar dan berdimensi tinggi.

---

## 1.4 Decision Trees

Decision Tree membuat keputusan menggunakan struktur seperti pohon. Setiap node berisi pertanyaan atau kondisi tertentu, kemudian proses dilanjutkan sampai mencapai leaf yang berisi hasil prediksi.

Pada praktiknya saya menggunakan dataset **two moons** dan mencoba mengatur parameter `max_depth`.

Dari eksperimen tersebut saya memahami bahwa semakin kompleks pohon yang dibuat, model dapat semakin mengikuti pola data training. Oleh karena itu, pengaturan kedalaman pohon penting untuk menghindari model yang terlalu kompleks.

Decision Tree memiliki kelebihan karena hasilnya relatif mudah divisualisasikan dan dijelaskan. Buku menggambarkan prosesnya sebagai rangkaian pertanyaan if/else yang dipelajari dari data menggunakan supervised learning.

---

## 1.5 Ensembles of Decision Trees — Random Forest

Random Forest merupakan ensemble yang menggunakan beberapa decision tree, kemudian menggabungkan hasil dari tree-tree tersebut.

Dalam praktiknya saya mencoba **Random Forest** dan melihat bagaimana penggunaan banyak tree dapat menghasilkan model yang lebih kuat dibandingkan hanya menggunakan satu decision tree.

Saya memahami bahwa kelebihan utama Random Forest adalah lebih stabil terhadap perubahan data dibandingkan satu decision tree. Namun, modelnya menjadi lebih sulit untuk dijelaskan secara langsung karena terdiri dari banyak tree.

Buku juga menjelaskan Random Forest sebagai model yang kuat dan robust serta tidak memerlukan scaling data.

---

## 1.6 Kernelized Support Vector Machines (SVM)

Kernelized SVM digunakan untuk mencari batas pemisah antar kelas, termasuk ketika hubungan antar kelas tidak dapat dipisahkan dengan garis lurus.

Dalam praktiknya saya menggunakan **RBF kernel** dan mencoba parameter seperti `C` dan `gamma`.

Dari percobaan tersebut saya memahami bahwa parameter SVM sangat berpengaruh terhadap bentuk decision boundary. `C` dan `gamma` dapat membuat model menjadi lebih sederhana atau lebih kompleks.

SVM merupakan algoritma yang cukup powerful, tetapi membutuhkan perhatian terhadap scaling data dan pemilihan parameter.

---

## 1.7 Neural Networks / Deep Learning

Neural Network menggunakan beberapa neuron yang disusun dalam layer untuk mempelajari pola dari data. Pada praktiknya saya menggunakan **MLPClassifier** dan mencoba konfigurasi jumlah hidden layer, jumlah neuron, fungsi aktivasi, serta nilai `alpha`.

Dari percobaan ini saya memahami bahwa Neural Network memiliki kemampuan untuk mempelajari pola yang lebih kompleks dibandingkan model sederhana. Namun, semakin kompleks arsitekturnya, semakin penting pemilihan parameter dan preprocessing data.

Saya juga melihat bahwa perubahan jumlah hidden layer dan neuron dapat mengubah decision boundary yang dihasilkan model. Buku menjelaskan bahwa neural networks mampu membangun model yang sangat kompleks, tetapi sensitif terhadap scaling data dan pemilihan parameter.

---

# 🔍 2. Unsupervised Learning

Berbeda dengan supervised learning, **Unsupervised Learning tidak menggunakan target/label sebagai acuan utama dalam proses pembelajaran**.

Tujuannya bukan memprediksi label yang sudah diketahui, tetapi menemukan struktur, pola, representasi, atau kelompok yang terdapat di dalam data.

Pada Chapter 3, materi unsupervised learning mencakup beberapa pendekatan, yaitu **dimensionality reduction, feature extraction, manifold learning, dan clustering**. Buku menempatkan PCA, NMF, t-SNE, k-Means, Agglomerative Clustering, dan DBSCAN dalam bagian ini.

### Model yang Dipelajari

| No. | Model                    | Fokus Pembelajaran                 |
| --- | ------------------------ | ---------------------------------- |
| 1   | PCA                      | Reduksi dimensi                    |
| 2   | NMF                      | Ekstraksi/representasi fitur       |
| 3   | t-SNE                    | Visualisasi data berdimensi tinggi |
| 4   | k-Means                  | Clustering berbasis centroid       |
| 5   | Agglomerative Clustering | Hierarchical clustering            |
| 6   | DBSCAN                   | Clustering berdasarkan kepadatan   |

---

## 2.1 Principal Component Analysis (PCA)

PCA digunakan untuk melakukan **reduksi dimensi** dengan mengubah data menjadi representasi baru yang lebih sederhana.

Dari praktik yang dilakukan, saya memahami bahwa PCA dapat membantu mengurangi jumlah fitur sehingga data yang awalnya memiliki banyak dimensi dapat divisualisasikan atau diproses dengan lebih sederhana.

Hal penting yang saya pahami adalah PCA tidak bertujuan melakukan klasifikasi. PCA lebih berfokus pada menemukan representasi data yang lebih ringkas.

---

## 2.2 Non-negative Matrix Factorization (NMF)

NMF merupakan metode yang digunakan untuk mendapatkan representasi data dengan komponen yang bernilai non-negatif.

Dari praktiknya, saya memahami bahwa NMF dapat digunakan untuk menemukan pola atau komponen yang membentuk data. Karena hasil komponennya tidak negatif, representasi yang dihasilkan dapat lebih mudah diinterpretasikan pada beberapa jenis data.

Berbeda dengan model supervised, NMF tidak membutuhkan target kelas untuk menghasilkan representasi tersebut.

---

## 2.3 t-SNE

t-SNE digunakan terutama untuk **visualisasi data berdimensi tinggi** menjadi ruang dengan dimensi lebih rendah, biasanya dua dimensi.

Pada praktiknya saya menggunakan **digits dataset** dan melakukan `fit_transform()` untuk mendapatkan representasi dua dimensi.

Hasil visualisasi membantu saya melihat bahwa data yang memiliki karakteristik mirip cenderung berada berdekatan pada representasi yang dihasilkan. Namun, saya juga memahami bahwa t-SNE lebih ditujukan untuk eksplorasi dan visualisasi daripada digunakan sebagai model prediksi.

---

## 2.4 k-Means Clustering

k-Means merupakan algoritma clustering yang mengelompokkan data berdasarkan kedekatannya terhadap sejumlah pusat cluster atau **centroid**.

Dalam praktiknya saya menentukan jumlah cluster, kemudian menjalankan proses clustering dan memvisualisasikan hasilnya.

Hal yang saya pahami adalah k-Means membutuhkan jumlah cluster yang ditentukan terlebih dahulu. Karena itu, pemilihan jumlah cluster menjadi salah satu hal penting ketika menggunakan algoritma ini.

---

## 2.5 Agglomerative Clustering

Agglomerative Clustering merupakan metode **hierarchical clustering** yang membangun kelompok secara bertahap.

Pada praktiknya saya menggunakan `AgglomerativeClustering` dengan jumlah cluster tertentu dan membandingkan hasil pengelompokan dengan data awal.

Dari eksperimen tersebut saya memahami bahwa algoritma ini memiliki pendekatan yang berbeda dari k-Means. Jika k-Means berpusat pada centroid, Agglomerative Clustering membangun struktur cluster secara hierarki.

---

## 2.6 DBSCAN

DBSCAN merupakan algoritma clustering yang menggunakan konsep **kepadatan data**.

Dalam praktiknya saya mencoba DBSCAN dan mengamati bagaimana data dapat dikelompokkan berdasarkan area yang memiliki kepadatan tertentu. Salah satu hal yang menarik adalah DBSCAN dapat mengenali data yang dianggap sebagai **noise/outlier**.

Hal tersebut membuat DBSCAN berbeda dari k-Means karena DBSCAN tidak harus selalu memaksa setiap titik data masuk ke salah satu cluster.

---

# ⚖️ 3. Perbandingan Supervised dan Unsupervised Learning

| Aspek             | Supervised Learning                                                                | Unsupervised Learning                                            |
| ----------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Label/Target      | Ada                                                                                | Tidak ada                                                        |
| Tujuan            | Prediksi                                                                           | Menemukan pola/struktur                                          |
| Contoh tugas      | Classification & Regression                                                        | Clustering & Dimensionality Reduction                            |
| Evaluasi          | Dapat menggunakan target sebenarnya                                                | Lebih bergantung pada struktur/hasil clustering atau visualisasi |
| Model yang dicoba | KNN, Linear Models, Naive Bayes, Decision Tree, Random Forest, SVM, Neural Network | PCA, NMF, t-SNE, k-Means, Agglomerative, DBSCAN                  |
| Hasil utama       | Prediksi kelas atau nilai                                                          | Kelompok atau representasi data                                  |

---

# 📊 4. Analisis Hasil Praktikum

Setelah mencoba seluruh algoritma di Google Colab, saya memahami bahwa **tidak ada satu algoritma yang selalu cocok untuk semua jenis dataset**.

Pada supervised learning, keberadaan label membuat proses pembelajaran lebih terarah. Model dapat dibandingkan menggunakan hasil prediksi terhadap target sebenarnya. Namun, model tetap perlu diuji pada data yang belum pernah dilihat agar dapat diketahui kemampuan generalisasinya.

Saya juga memahami bahwa kompleksitas model memiliki pengaruh terhadap hasil. Model sederhana seperti Linear Models relatif mudah dipahami, sedangkan Random Forest, SVM, dan Neural Network dapat menangani pola yang lebih kompleks tetapi membutuhkan perhatian lebih terhadap parameter dan karakteristik data.

Pada unsupervised learning, tantangannya berbeda karena tidak tersedia label sebagai jawaban benar. Oleh karena itu, hasil yang diperoleh lebih banyak digunakan untuk memahami struktur data. Misalnya, k-Means, Agglomerative Clustering, dan DBSCAN sama-sama melakukan clustering tetapi menggunakan pendekatan yang berbeda. Sementara itu, PCA, NMF, dan t-SNE lebih berfokus pada representasi atau eksplorasi data.

Dari keseluruhan praktikum, saya mendapatkan pemahaman bahwa **pemilihan algoritma harus disesuaikan dengan tujuan, karakteristik dataset, jumlah data, jumlah fitur, dan pola yang terdapat pada data**. Buku juga menekankan bahwa menerapkan algoritma secara langsung tanpa memahami asumsi model, representasi data, dan parameter dapat menghasilkan model yang kurang baik.

---

# 💡 5. Insight yang Didapat

Beberapa hal penting yang saya dapatkan setelah melakukan praktik adalah:

1. **Supervised dan unsupervised memiliki tujuan yang berbeda.** Supervised lebih berorientasi pada prediksi, sedangkan unsupervised digunakan untuk menemukan pola atau struktur.

2. **Dataset sangat menentukan algoritma yang digunakan.** Algoritma yang bagus pada satu dataset belum tentu memberikan hasil yang sama pada dataset lain.

3. **Parameter model memiliki pengaruh besar.** Contohnya nilai K pada KNN, `alpha` pada Ridge dan Neural Network, `max_depth` pada Decision Tree, serta `C` dan `gamma` pada SVM.

4. **Preprocessing dan scaling penting untuk beberapa algoritma.** SVM dan Neural Network khususnya sensitif terhadap scaling data, sedangkan Decision Tree dan Random Forest tidak memerlukannya dalam cara yang sama.

5. **Visualisasi membantu memahami perilaku model.** Decision boundary pada supervised learning serta visualisasi hasil clustering dan t-SNE membuat konsep yang sebelumnya abstrak menjadi lebih mudah dipahami.

6. **Model yang kompleks tidak selalu menjadi pilihan pertama.** Dari pembelajaran ini saya memahami bahwa model sederhana dapat digunakan terlebih dahulu sebagai baseline sebelum mencoba model yang lebih kompleks. Buku juga menyarankan memulai dari model sederhana kemudian mempertimbangkan model yang lebih kompleks setelah karakteristik data lebih dipahami.

---

# 🚀 6. Kesimpulan

Pembelajaran 13 algoritma Machine Learning melalui Google Colab memberikan pemahaman dasar mengenai dua pendekatan utama, yaitu **Supervised Learning** dan **Unsupervised Learning**.

Supervised Learning yang dipelajari terdiri dari **KNN, Linear Models, Naive Bayes, Decision Trees, Random Forest, Kernelized SVM, dan Neural Networks**. Model-model tersebut digunakan ketika tersedia target atau label dan dapat digunakan untuk tugas klasifikasi maupun regresi.

Sementara itu, Unsupervised Learning terdiri dari **PCA, NMF, t-SNE, k-Means, Agglomerative Clustering, dan DBSCAN**. Algoritma tersebut digunakan untuk memahami struktur data tanpa bergantung pada label target.

Melalui implementasi di Google Colab, saya tidak hanya mempelajari sintaks Python dan penggunaan `scikit-learn`, tetapi juga mulai memahami bahwa Machine Learning bukan hanya tentang menjalankan model. Hal yang lebih penting adalah **memahami karakteristik data, tujuan permasalahan, parameter model, hasil eksperimen, serta alasan memilih suatu algoritma**.

---

## 📚 Referensi

* Müller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.
* Dokumentasi dan implementasi algoritma menggunakan Python dan `scikit-learn`.
