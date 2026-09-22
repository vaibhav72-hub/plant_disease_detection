TOMATO PLANT DISEASE DETECTION USING DEEP LEARNING
================================================

This project detects diseases in tomato plant leaves using a
Convolutional Neural Network (CNN) trained on the PlantVillage dataset.
It provides a web interface using Flask to upload an image and view
disease details.

------------------------------------------------
1. PROJECT STRUCTURE
------------------------------------------------

TOMATO_PLANT/
│
├── dataset/
│   └── PlantVillage/
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       ├── Tomato___Late_blight/
│       ├── Tomato___Leaf_Mold/
│       ├── Tomato___Septoria_leaf_spot/
│       ├── Tomato___Spider_mites/
│       ├── Tomato___Target_Spot/
│       ├── Tomato___Tomato_mosaic_virus/
│       ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│       └── Tomato___healthy/
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── diseases/
│       ├── bacterial_spot.html
│       ├── early_blight.html
│       ├── late_blight.html
│       ├── leaf_mold.html
│       ├── septoria.html
│       ├── spider_mites.html
│       ├── target_spot.html
│       ├── mosaic_virus.html
│       ├── yellow_leaf_curl.html
│       └── healthy.html
│
├── app.py
├── train_model.py
├── model.h5
├── requirements.txt
└── README.txt

------------------------------------------------
2. SYSTEM REQUIREMENTS
------------------------------------------------

• Python 3.9 or later (Recommended: Python 3.10)
• Windows / Linux / macOS
• Minimum 8 GB RAM recommended
• GPU optional (CPU also works)

------------------------------------------------
3. CREATE VIRTUAL ENVIRONMENT (OPTIONAL)
------------------------------------------------

Windows:
> python -m venv venv
> venv\Scripts\activate

Linux / macOS:
> python3 -m venv venv
> source venv/bin/activate

------------------------------------------------
4. INSTALL DEPENDENCIES
------------------------------------------------

Run this command in the project root folder:

> pip install -r requirements.txt

------------------------------------------------
5. TRAIN THE MODEL
------------------------------------------------

This step trains the CNN model and also prints
the total training time.

Run:

>   ``  python train_model.py
2
Output includes:
• Training accuracy & loss
• Validation accuracy & loss
• Total training time (seconds & minutes)
• Saved model file: model.h5

NOTE:
Training may take several minutes depending on
system performance and dataset size.

------------------------------------------------
6. RUN THE FLASK WEB APPLICATION
------------------------------------------------

After training is complete, start the web app:

> python app.py

You should see:
* Running on http://127.0.0.1:5000/

------------------------------------------------
7. USE THE WEB APPLICATION
------------------------------------------------

1. Open browser
2. Go to:
   http://127.0.0.1:5000/
3. Upload a tomato leaf image
4. Click "Predict"
5. View:
   • Predicted disease
   • Uploaded image
6. Click "View Disease Details"
7. See prevention and treatment info

------------------------------------------------
8. MODEL OUTPUT CLASSES
------------------------------------------------

• Bacterial Spot
• Early Blight
• Late Blight
• Leaf Mold
• Septoria Leaf Spot
• Spider Mites
• Target Spot
• Tomato Mosaic Virus
• Yellow Leaf Curl Virus
• Healthy

------------------------------------------------
9. IMPORTANT NOTES
------------------------------------------------

• Use clear leaf images for better accuracy
• Dataset folder names must not be changed
• Delete old images from static/uploads if needed
• Retrain model if dataset is updated

------------------------------------------------
10. FUTURE ENHANCEMENTS
------------------------------------------------

• Confidence percentage display
• Grad-CAM heatmaps
• MobileNet / Transfer Learning
• Video-based disease detection
• Cloud deployment (AWS / Azure)

------------------------------------------------
PROJECT STATUS
------------------------------------------------

✔ Fully functional
✔ Train + Predict
✔ Web UI integrated
✔ Ready for academic submission
✔ Resume & interview ready

------------------------------------------------



python -m pip install --upgrade pip
pip install tensorflow==2.10.1
pip install numpy==1.23.5 pillow flask opencv-python
pip install scipy
