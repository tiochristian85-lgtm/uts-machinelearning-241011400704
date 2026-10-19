# UTS Machine Learning - 241011400704

**Nama:** Christian Tio  
**NIM:** 241011400704  
**Mata Kuliah:** Machine Learning  
**Repo:** UTS-MACHINE-LEARNING-241011400704

### Deskripsi

Implementasi **Logistic Regression dari scratch** untuk Filter Spam Email:
$$P(Spam) = \\sigma(b + m\_1 \\cdot JumlahLink + m\_2 \\cdot KataPromo)$$

### 📁 Struktur

```
UTS-MACHINE-LEARNING-241011400704/
├── tugas\\\_UTS.ipynb
├── README.md
```

### 🚀 Hasil Utama (Bagian B)

**B2 - Training:** Log Loss turun dari 0.6931 -> 0.4557, grafik menunjukkan model belajar.

**B3 - Evaluasi:**

* Implementasi manual Accuracy, Confusion Matrix, Precision, Recall, F1
* **Insight:** FP (email penting dianggap spam) lebih merugikan daripada FN (spam lolos)

**B4 - Decision Boundary:**

* Persamaan: x2 = -(b + m1\*x1)/m2
* Visualisasi contourf probabilitas + garis boundary + penanda FP/FN

**B5 - Threshold:**

* Uji 0.3, 0.5, 0.7
* Threshold 0.3: Recall tinggi, FP banyak
* Threshold 0.7: Precision tinggi, FP sedikit
* **Pilihan: 0.7** untuk meminimalisir FP

**B6 - Prediksi Baru:**

* E1(1,2)=NOT SPAM, E2(5,6)=Borderline, E3(9,10)=SPAM, E4(2,11)=SPAM

### 📝 Bagian C - Konsep

**C1:** Disebut regression karena menghitung z=b+m1x1+m2x2, tapi jadi klasifikasi setelah dilewatkan sigmoid jadi probabilitas 0-1.
**C2:** Normal equation = solusi langsung (O(n^3)), bagus untuk fitur sedikit. Gradient descent = iteratif, wajib untuk fitur >10k.
**C3:** Evaluasi di test untuk cek generalisasi. Overfitting = hafal training, jelek di test.
**C4:** Linear pakai MSE karena output kontinu. Logistic pakai Log Loss karena output probabilitas kelas, MSE jadi non-convex.
**C5:** Beda hasil karena random\_state split, inisialisasi bobot, learning rate, dan jumlah iterasi.

### ⭐ Bonus - sklearn

* **LinearRegression:** Hasil manual vs sklearn hampir sama (tanpa regularisasi).
* **LogisticRegression:** Beda karena sklearn default penalty='l2', C=1.0 (ada regularisasi L2 yang mengecilkan koefisien). Jika set penalty=None atau C=1e6, hasil mirip manual.
* Penyebab beda adalah regularisasi bawaan sklearn.

### ▶️ Cara Run

```bash
pip install numpy matplotlib pandas scikit-learn
jupyter notebook tugas\\\_UTS.ipynb
```

Run All di Jupyter, Python 3.14.3



