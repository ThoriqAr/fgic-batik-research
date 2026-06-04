# fgic-batik-research

`fgic-batik-research` merupakan sebuah repository yang digunakan untuk melakukan experiment fine grained image classification pada motif batik. Pada file `README.md` ini berisikan requirements yang dibutuhkan, cara menyalin repository dari remote ke local, serta step by step penginstallan dependecy-nya supaya repository bisa dijalankan.

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

## 6. Verify Installation

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