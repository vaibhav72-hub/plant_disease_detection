# 🌿 Multi-Plant Disease Detection System using Deep Learning

An end-to-end Computer Vision and Deep Learning web application designed to automatically identify crops and detect diseases across **38 classes** in **14 different agricultural plants** using the **PlantVillage dataset**.

---

## 🌟 Key Features

- **Multi-Crop Detection**: Simultaneously classifies plant species and pathology across Tomato, Potato, Bell Pepper, Corn, Apple, Grape, Peach, Cherry, Strawberry, Blueberry, Soybean, Squash, and Citrus.
- **Transfer Learning with MobileNetV2**: High-accuracy deep learning architecture pre-trained on ImageNet with custom classification head for fast convergence.
- **Automated Dataset Downloader**: One-click download and directory preparation via `download_dataset.py` using `kagglehub`.
- **Comprehensive Disease Knowledge Base**: Detailed diagnosis, symptoms, organic/cultural remedies, chemical treatments, and direct product links for each condition.
- **Modern Responsive Web Interface**: Glassmorphism UI with live drag-and-drop image preview, diagnostic confidence meter, and clean report layout.

---

## 📁 Project Structure

```text
plant_disease_detection/
├── dataset/
│   └── PlantVillage/                  # 38 plant & disease subfolders
├── static/
│   ├── css/
│   │   └── style.css                  # Modern UI styles
│   └── uploads/                       # Uploaded leaf images for diagnosis
├── templates/
│   ├── home.html                      # Landing page with drag-and-drop upload
│   └── predict.html                   # Detailed pathology report
├── app.py                             # Flask web server & inference engine
├── download_dataset.py                # Automated PlantVillage dataset downloader
├── train_model.py                     # MobileNetV2 / CNN training pipeline
├── disease_data.json                  # Knowledge base for all 38 classes
├── class_indices.json                 # Label-to-class index mapping
├── model.h5                           # Trained neural network weights
└── requirements.txt                   # Project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Multi-Crop Dataset

To automatically download the PlantVillage dataset (~1.5 GB, 38 classes) and prepare the directory:

```bash
python download_dataset.py
```

*Note: You can verify existing datasets anytime by running `python download_dataset.py --check`.*

### 4. Train the Model

Train the MobileNetV2 transfer learning model (or lightweight CNN):

```bash
# Train using MobileNetV2 (Recommended for high accuracy >95%)
python train_model.py --architecture mobilenet --epochs 10 --batch-size 32

# Or train using a custom CNN
python train_model.py --architecture cnn --epochs 15
```

The script will automatically:
- Discover all classes in `dataset/PlantVillage/`
- Apply data augmentation (flips, rotations, zoom)
- Train with early stopping and learning rate scheduling
- Save the trained weights to `model.h5` and class list to `class_indices.json`

### 5. Launch the Web Application

```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

Upload any crop leaf image to receive instant diagnosis, confidence score, and treatment recommendations!

---

## 🔬 Supported Crops & Pathologies (38 Classes)

| Crop | Supported Conditions / Diseases |
| :--- | :--- |
| **Tomato** | Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Bell Pepper** | Bacterial Spot, Healthy |
| **Corn (Maize)** | Cercospora / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Spot), Healthy |
| **Peach** | Bacterial Spot, Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Strawberry** | Leaf Scorch, Healthy |
| **Blueberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Orange / Citrus**| Huanglongbing (Citrus Greening) |
| **Raspberry** | Healthy |
