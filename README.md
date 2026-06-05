# fgic-batik-research

`fgic-batik-research` merupakan sebuah repository yang digunakan untuk melakukan eksperimen fine grained image classification pada motif batik. Pada file `README.md` ini berisikan requirements yang dibutuhkan, cara menyalin repository dari remote ke local, serta step by step penginstallan dependecy-nya supaya repository bisa dijalankan.

# Requirements

## Recommended Environment

Project ini dikembangkan dan diuji menggunakan:

* Python 3.10.11
* TensorFlow 2.21.0
* Keras 3.12.1
* NumPy 1.26.4

Menggunakan versi Python selain 3.10.x tidak direkomendasikan karena dapat menyebabkan konflik dependency pada TensorFlow dan library pendukung lainnya.

## Dependency Files

Repository menyediakan dua file dependency:

### 1. requirements.txt

Berisi dependency utama yang diperlukan untuk menjalankan project.

Digunakan untuk instalasi standar:

```bash
pip install -r requirements.txt
```

### 2. requirements-lock.txt

Berisi snapshot lengkap environment hasil `pip freeze`.

Digunakan untuk reproduksi eksperimen secara identik dengan environment pengembangan:

```bash
pip install -r requirements-lock.txt
```

File ini hanya digunakan apabila diperlukan reproduksibilitas penuh terhadap hasil eksperimen.

# Installation

## 1. Clone Repository

Clone repository terlebih dahulu dari remote ke lokal:

```bash
git clone <repository-url>
```

Buka repository yang sudah di clone:

```bash
cd <repository-name>
```

## 2. Create Virtual Environment

Sebelum membuat `.venv` pastikan root pada terminal sudah mengarah ke repository yang sudah di clone, contohnya `<root_directory>\fgic-batik-research` lalu jalankan perintah berikut.

Windows:

```bash
py -3.10 -m venv .venv
```

Linux / MacOS:

```bash
python3.10 -m venv .venv
```

## 3. Activate Virtual Environment

Aktifkan `.venv` yang sudah dibuat menggunakan perintah berikut.

Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / MacOS:

```bash
source .venv/bin/activate
```

## 4. Upgrade pip

Upgrade pip terlebih dahulu menggunakan perintah berikut.

```bash
python -m pip install --upgrade pip
```

## 5. Install Dependencies

Pilih salah satu sumber dependency yang akan di install.

Standard installation:

```bash
pip install -r requirements.txt
```

atau untuk reproduksi penuh:

```bash
pip install -r requirements-lock.txt
```

## 6. Verify Installation Dependencies

Verifikasi hasil install dependency utama menggunakan perintah berikut.

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
python -c "import lime; print('lime ok')"
python -c "import quantus; print('quantus ok')"
python -c "import tf_keras_vis; print('tf_keras_vis ok')"
```

## 7. Dependency Validation

Pastikan seluruh dependency terpasang dengan benar:

```bash
pip check
```

Output yang diharapkan:

```text
No broken requirements found.
```

Jika muncul konflik dependency, hapus virtual environment dan lakukan instalasi ulang menggunakan Python 3.10.11.

## 8. Install Project Package

Langkah ini diperlukan agar module di folder `src` dapat di-import dari notebook maupun script.

```bash
pip install -e .
```

## 9. Verifikasi Project Package

Pastikan project package terinstall dengan benar:

```bash
pip show fgic-batik-research
```

Output yang diharapkan mengandung baris berikut:

```text
Editable project location: <root_directory>\fgic-batik-research
```

# Dataset Preparation

Repository ini menggunakan metadata dataset yang telah ditentukan sebelumnya untuk memastikan seluruh eksperimen dapat direproduksi secara konsisten.

Metadata berikut telah tersedia di repository:

```text
dataset/metadata/
├── class_mapping.json
├── train.csv
├── val.csv
└── test.csv
```

File tersebut merupakan konfigurasi resmi yang digunakan pada seluruh eksperimen penelitian.

Sebelum melakukan eksperimen, ikuti tahapan dataset preparation berikut:

## 1. Download Dataset

Unduh dataset terlebih dahulu pada link berikut

https://drive.google.com/drive/folders/1pBQBrUh78Ckn-YzBx6x5YXrSzCVikxIO?usp=sharing

lalu letakkan pada direktori berikut:

```text
dataset/raw/
```

Struktur yang diharapkan:

```text
dataset/raw/
├── class_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Setiap folder merepresentasikan satu kelas batik.

---

## 2. Standardize Dataset

Lakukan standarisasi dataset untuk:

* menyamakan format gambar menjadi JPG
* menormalkan penamaan file
* menghasilkan dataset yang konsisten dan mudah direproduksi

Jalankan:

```bash
python src/data/standardize_dataset.py
```

Output:

```text
dataset/processed/
├── class_1/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── class_2/
│   ├── image_0001.jpg
│   └── ...
└── ...
```

Selain itu akan dihasilkan:

```text
dataset/reports/
├── dataset_info.json
└── file_mapping.csv
```

---

## 3. Verify Dataset

Pastikan jumlah gambar pada dataset hasil standarisasi sesuai dengan dataset penelitian.

Gunakan laporan untuk melakukan pemeriksaan:

```text
dataset/reports/dataset_info.json
```

Output yang diharapkan:

```json
{
    "num_classes": 20,
    "total_images": 3000,
    "image_format": "jpg"
}
```

---

## 4. Run Experiments

Setelah dataset berhasil distandarisasi, seluruh eksperimen dapat langsung menggunakan metadata yang tersedia pada:

```text
dataset/metadata/
```

Tidak diperlukan pembuatan split ulang maupun class mapping ulang.

# Advanced Usage

Bagian ini hanya diperlukan apabila dataset berubah atau pengguna ingin membuat metadata baru.

## Peringatan

Perintah pada bagian ini dapat menimpa metadata yang sudah ada.

Jika dataset yang digunakan tidak identik dengan dataset penelitian asli, maka hasil split yang dihasilkan dapat berbeda dan tidak lagi merepresentasikan konfigurasi eksperimen yang digunakan pada penelitian ini.

## 1. Generate Class Mapping

Buat mapping kelas yang akan digunakan selama penelitian.

Jalankan:

```bash
python src/data/generate_class_mapping.py
```

Output:

```text
dataset/metadata/class_mapping.json
```

Contoh:

```json
{
    "batik_betawi": 0,
    "batik_bokor_kencono": 1
}
```

---

## 2. Create Dataset Split

Buat pembagian train, validation, dan test set menggunakan stratified split.

Jalankan:

```bash
python src/data/create_split.py
```

Output:

```text
dataset/metadata/train.csv
dataset/metadata/val.csv
dataset/metadata/test.csv
```

Setiap file CSV berisi:

```text
filepath,label,class_id
```

Contoh:

```csv
filepath,label,class_id
batik_betawi/batik_betawi_0001.jpg,batik_betawi,0
batik_bokor_kencono/batik_bokor_kencono_00001.jpg,batik_bokor_kencono,1
```