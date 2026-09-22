from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import os
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# model = load_model("model.h5")
class MockModel:
    def predict(self, *args, **kwargs):
        # Return a mock prediction array where class 1 (early_blight) has highest probability
        return np.array([[0.1, 0.8, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
model = MockModel()


try:
    with open("class_indices.json", "r") as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    CLASS_NAMES = [
        "bacterial_spot",
        "early_blight",
        "late_blight",
        "leaf_mold",
        "septoria",
        "spider_mites",
        "target_spot",
        "mosaic_virus",
        "yellow_leaf_curl",
        "healthy"
    ]

disease_info = {
    "bacterial_spot": {
        "name": "Bacterial Spot",
        "description": "Bacterial spot is caused by Xanthomonas bacteria. It causes dark water-soaked spots on leaves and fruits.",
        "treatment": "Remove infected plant parts immediately. Avoid overhead watering to reduce moisture on leaves. Apply copper-based fungicides early in the season to prevent outbreaks.",
        "pesticides": [
            {"name": "Bonide Copper Fungicide", "link": "https://www.amazon.com/Bonide-811-Copper-Fungicide-4-Pound/dp/B000UJMWJ2"},
            {"name": "Southern Ag Liquid Copper", "link": "https://www.amazon.com/Southern-Ag-Liquid-Fungicide-16oz/dp/B004GD5V4G"}
        ]
    },
    "early_blight": {
        "name": "Early Blight",
        "description": "Caused by the fungus Alternaria solani. Symptoms include concentric ring spots (target-like) on lower leaves.",
        "treatment": "Improve air circulation and avoid overhead watering. Mulch around the base of the plant to prevent soil splash. Use a fungicide containing chlorothalonil or mancozeb.",
        "pesticides": [
            {"name": "Daconil Fungicide", "link": "https://www.amazon.com/GardenTech-Daconil-Fungicide-Concentrate-16-Ounce/dp/B000RUJZS6"},
            {"name": "Bonide Mancozeb", "link": "https://www.amazon.com/Bonide-862-Mancozeb-Flowable-Concentrate/dp/B000UJQY3S"}
        ]
    },
    "late_blight": {
        "name": "Late Blight",
        "description": "A serious disease caused by Phytophthora infestans. It causes large, dark, greasy-looking spots on leaves and stems.",
        "treatment": "This disease spreads rapidly in cool, wet weather. Remove specific infected leaves or entire plants if severe. Fungicides with copper or chlorothalonil can be effective as preventatives.",
        "pesticides": [
            {"name": "Monterey Liqui-Cop", "link": "https://www.amazon.com/Monterey-LG-3415-Liqui-Cop-Fungicide/dp/B000BWZ9U8"},
            {"name": "Daconil Ready-to-use", "link": "https://www.amazon.com/GardenTech-Daconil-Fungicide-Ready-Use/dp/B000RUI5SY"}
        ]
    },
    "leaf_mold": {
        "name": "Leaf Mold",
        "description": "Caused by Passalora fulva. It appears as pale green or yellow spots on the upper leaf surface and olive-green velvet mold on the underside.",
        "treatment": "High humidity is the main driver. Improve airflow by pruning and spacing plants. Avoid wetting leaves. Fungicides can be used if severe.",
        "pesticides": [
            {"name": "Daconil Fungicide", "link": "https://www.amazon.com/GardenTech-Daconil-Fungicide-Concentrate-16-Ounce/dp/B000RUJZS6"},
            {"name": "Neem Oil", "link": "https://www.amazon.com/Organic-Neem-Bliss-100-Pressed/dp/B0716JF8MB"}
        ]
    },
    "septoria": {
        "name": "Septoria Leaf Spot",
        "description": "Caused by Septoria lycopersici. Circular spots with dark borders and gray centers appear on lower leaves.",
        "treatment": "Remove lower infected leaves. Mulch to prevent soil splashing spores onto the plant. Water at the base. Apply fungicides if necessary.",
        "pesticides": [
            {"name": "Bonide Copper Fungicide", "link": "https://www.amazon.com/Bonide-811-Copper-Fungicide-4-Pound/dp/B000UJMWJ2"},
            {"name": "Spectracide Immunox", "link": "https://www.amazon.com/Spectracide-Immunox-Multi-Purpose-Fungicide-Concentrate/dp/B0035H0RA8"}
        ]
    },
    "spider_mites": {
        "name": "Two Spotted Spider Mite",
        "description": "Tiny pests that suck sap from leaves, causing yellow stippling. You might see fine webbing on the plant.",
        "treatment": "For control, use selective products whenever possible. Selective products which have worked well in the field include: bifenazate (Acramite): Group UN, a long residual nerve poison abamectin (Agri-Mek): Group 6, derived from a soil bacterium spirotetramat (Movento): Group 23, mainly affects immature stages spiromentifen (Oberon 2SC): Group 23, mainly affects immature stages OMRI-listed products include: insecticidal soap (M-Pede) neem oil (Trilogy) soybean oil (Golden Pest Spray Oil) With most miticides (excluding bifenazate), make 2 applications, approximately 5-7 days apart, to help control immature mites that were in the egg stage and protected during the first application. Alternate between products after 2 applications to help prevent or delay resistance.",
        "pesticides": [
            {"name": "Abamectin", "link": "https://www.biglittlefarms.com/abamectin-insecticide.html"},
            {"name": "Spiromesifen", "link": "https://www.amazon.in/Oberon-Bayer-Spiromesifen-Insecticide-Acaricide/dp/B07L56F5L1"}
        ]
    },
    "target_spot": {
        "name": "Target Spot",
        "description": "Fungal disease causing brown to black spots with concentric rings, similar to Early Blight but often attacks fruit as well.",
        "treatment": "Remove infected plant debris. Improve airflow. Apply fungicides effective against Corynespora cassiicola.",
        "pesticides": [
            {"name": "Daconil Fungicide", "link": "https://www.amazon.com/GardenTech-Daconil-Fungicide-Concentrate-16-Ounce/dp/B000RUJZS6"},
            {"name": "Copper Fungicide", "link": "https://www.amazon.com/Bonide-811-Copper-Fungicide-4-Pound/dp/B000UJMWJ2"}
        ]
    },
    "mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "Viral disease causing mottled (mosaic) yellow and green patterns on leaves. Leaves may curl or become distorted.",
        "treatment": "There is no cure. Remove and destroy infected plants to prevent spread. Wash hands after using tobacco products, as they can carry the virus. Control aphids.",
        "pesticides": [
            {"name": "None (Remove Plant)", "link": "#"},
            {"name": "Control Aphids with Neem Oil", "link": "https://www.amazon.com/Organic-Neem-Bliss-100-Pressed/dp/B0716JF8MB"}
        ]
    },
    "yellow_leaf_curl": {
        "name": "Yellow Leaf Curl Virus",
        "description": "Transmitted by whiteflies. Leaves curl upward and turn yellow. Plants become stunted and stop producing fruit.",
        "treatment": "No cure for the virus. Control whiteflies with sticky traps and insecticides. Remove infected plants.",
        "pesticides": [
            {"name": "Yellow Sticky Traps", "link": "https://www.amazon.com/Garsum-Sticky-Traps-Flying-Insects/dp/B07ZSC4CK1"},
            {"name": "Monterey Garden Insect Spray", "link": "https://www.amazon.com/Monterey-LG6135-Garden-Insect-Spinosad/dp/B002BP12LI"}
        ]
    },
    "healthy": {
        "name": "Healthy Plant",
        "description": "Your plant looks healthy! No diseases detected.",
        "treatment": "Continue regular care: consistent watering, proper fertilization, and monitoring for pests. Preventative measures like mulching are always good.",
        "pesticides": [
            {"name": "Organic Fertilizer", "link": "https://www.amazon.com/Espoma-PT18-Plant-Tone-Organic-Fertilizer/dp/B003SOTNMY"},
            {"name": "Neem Oil (Preventative)", "link": "https://www.amazon.com/Organic-Neem-Bliss-100-Pressed/dp/B0716JF8MB"}
        ]
    }
}

def predict_disease(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return CLASS_NAMES[class_index]

@app.route('/')
def home():
    return render_template('home.html')







@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        disease = predict_disease(path)

        disease_details = disease_info.get(disease, {
            "name": disease.replace("_", " ").title(),
            "description": "Information not available for this plant/disease.",
            "treatment": "No treatment info available.",
            "pesticides": []
        })

        return render_template(
            "predict.html",
            disease_name=disease,
            image_name=file.filename,
            details=disease_details
        )

    return redirect('/')



#  IMPORTANT: MOBILE ACCESS ENABLED
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
