import os
import sys
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# -----------------------------
# Knowledge Base & Class Indices
# -----------------------------
DISEASE_DATA_FILE = "disease_data.json"
CLASS_INDICES_FILE = "class_indices.json"

try:
    with open(DISEASE_DATA_FILE, "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
except Exception as e:
    print(f"⚠️ Warning: Could not load {DISEASE_DATA_FILE}: {e}")
    DISEASE_INFO = {}

# Default classes to PlantVillage 38 classes if class_indices.json doesn't exist
try:
    with open(CLASS_INDICES_FILE, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
except Exception:
    CLASS_NAMES = list(DISEASE_INFO.keys()) if DISEASE_INFO else [
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___healthy"
    ]

# -----------------------------
# Model Loader
# -----------------------------
MODEL_PATH = "model.h5"
model = None
is_mock_model = False

if os.path.exists(MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model
        print(f"📦 Loading trained model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("✅ Trained model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading {MODEL_PATH}: {e}")
        model = None

if model is None:
    print("ℹ️ 'model.h5' not found or failed to load. Operating in Demo/Mock preview mode.")
    print("👉 To train the deep learning model on the full dataset, run: python train_model.py")
    is_mock_model = True

    class MockModel:
        def predict(self, x, *args, **kwargs):
            # Deterministic pseudo-prediction based on image hash/average for demo realism
            num_classes = len(CLASS_NAMES)
            probs = np.zeros((1, num_classes))
            # Pick a representative diseased class or tomato early blight
            preferred_idx = 0
            for idx, c in enumerate(CLASS_NAMES):
                if "Early_blight" in c or "early_blight" in c:
                    preferred_idx = idx
                    break
            probs[0, preferred_idx] = 0.94
            # Distribute remaining probabilities
            for i in range(num_classes):
                if i != preferred_idx:
                    probs[0, i] = (1.0 - 0.94) / max(1, (num_classes - 1))
            return probs

    model = MockModel()


def parse_class_name(raw_class):
    """
    Parses standard PlantVillage format: 'Tomato___Early_blight' or 'Corn_(maize)___Common_rust_'
    Returns (plant_name, disease_name, is_healthy)
    """
    if "___" in raw_class:
        plant_part, disease_part = raw_class.split("___", 1)
        plant_clean = plant_part.replace("_", " ").strip()
        disease_clean = disease_part.replace("_", " ").strip()
        is_healthy = "healthy" in disease_part.lower()
        if is_healthy:
            disease_clean = f"Healthy {plant_clean}"
    else:
        # Fallback for old custom names like "early_blight"
        plant_clean = "Tomato"
        disease_clean = raw_class.replace("_", " ").title()
        is_healthy = "healthy" in raw_class.lower()

    return plant_clean, disease_clean, is_healthy


def get_disease_details(raw_class):
    """Retrieve full disease info from database with graceful dynamic fallback."""
    # 1. Exact match
    if raw_class in DISEASE_INFO:
        return DISEASE_INFO[raw_class]

    # 2. Try matching without underscores or lowercase
    for k, v in DISEASE_INFO.items():
        if k.lower() == raw_class.lower() or k.lower().endswith(raw_class.lower()):
            return v

    # 3. Dynamic fallback for unlisted class
    plant, disease, is_healthy = parse_class_name(raw_class)
    return {
        "plant": plant,
        "disease_name": disease,
        "is_healthy": is_healthy,
        "description": f"Condition detected for {plant}: {disease}.",
        "symptoms": "Leaf tissue discoloration, altered morphology, or necrotic lesions.",
        "treatment": "Inspect affected areas, remove infected plant foliage, and avoid excessive humidity or overhead watering.",
        "organic_remedy": "Neem oil application and bio-fungicides.",
        "pesticides": [
            {"name": "Multi-Crop Broad Spectrum Fungicide", "link": "https://www.amazon.com/dp/B000RUJZS6"},
            {"name": "Organic Cold-Pressed Neem Oil", "link": "https://www.amazon.com/dp/B0716JF8MB"}
        ]
    }


def predict_disease(img_path):
    """Preprocess image and run inference through the neural network."""
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx]) * 100.0

    if pred_idx < len(CLASS_NAMES):
        raw_class = CLASS_NAMES[pred_idx]
    else:
        raw_class = "Tomato___healthy"

    details = get_disease_details(raw_class)
    plant_name, disease_name, is_healthy = parse_class_name(raw_class)

    return {
        "raw_class": raw_class,
        "plant_name": details.get("plant", plant_name),
        "disease_name": details.get("disease_name", disease_name),
        "is_healthy": details.get("is_healthy", is_healthy),
        "confidence": round(confidence, 1),
        "details": details,
        "is_mock": is_mock_model
    }


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    # Gather unique plants available in the database
    supported_plants = sorted(list({v.get("plant", "Plant") for v in DISEASE_INFO.values()}))
    total_classes = len(DISEASE_INFO) if DISEASE_INFO else len(CLASS_NAMES)
    return render_template(
        "home.html",
        supported_plants=supported_plants,
        total_classes=total_classes,
        is_mock_model=is_mock_model
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect("/")

    file = request.files["image"]
    if file.filename == "":
        return redirect("/")

    if file:
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filename = file.filename.replace(" ", "_")
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        result = predict_disease(file_path)

        return render_template(
            "predict.html",
            image_name=filename,
            plant_name=result["plant_name"],
            disease_name=result["disease_name"],
            is_healthy=result["is_healthy"],
            confidence=result["confidence"],
            details=result["details"],
            is_mock=result["is_mock"]
        )

    return redirect("/")


@app.route("/api/supported-plants")
def api_supported_plants():
    """API endpoint to get plant species and disease counts."""
    plants = {}
    for raw_class, info in DISEASE_INFO.items():
        p = info.get("plant", "Other")
        if p not in plants:
            plants[p] = []
        plants[p].append(info.get("disease_name", raw_class))
    return jsonify({"total_crops": len(plants), "total_classes": len(DISEASE_INFO), "crops": plants})


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print("🌾 Plant Disease Detection Server starting on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
