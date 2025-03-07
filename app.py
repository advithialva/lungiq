from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from pymongo import MongoClient
import cv2
import numpy as np
import os
import pickle
import gdown
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__,static_folder='static')

# MongoDB connection
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client['LungIQ']
contacts_collection = db['contact']

# Google Drive File IDs (Direct Download Links)
MODEL_ID = "1nf7gW58ecTytqk43EnjY397CTxcsaWOB"
ENCODER_ID = "1hkItbQXPJANGmPF3LCVRfpUgjl__3izB"

MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
ENCODER_URL = f"https://drive.google.com/uc?id={ENCODER_ID}"

# Paths for model and encoder (store in /tmp to persist in Render)
MODEL_PATH = "/tmp/CNN_Covid19_Xray_Version.h5"
ENCODER_PATH = "/tmp/Label_encoder.pkl"

# Ensure the /tmp directory exists
os.makedirs("/tmp", exist_ok=True)

# Download model and encoder if they don’t exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(ENCODER_PATH):
    print("Downloading label encoder...")
    gdown.download(ENCODER_URL, ENCODER_PATH, quiet=False)

# Ensure model files exist before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found at {ENCODER_PATH}")

# Load Model
model = load_model(MODEL_PATH)

# Load Label Encoder
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Path to store uploaded images
UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to process image and make predictions
def process_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    predictions = model.predict(image_input)
    
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]
    predicted_label = le.inverse_transform([predicted_index])[0]
    
    return predicted_label, confidence_score

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success_message = None
    
    if request.method == 'POST':
        # Retrieve form data
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject', 'No Subject')
        message = request.form.get('message')

        # Insert data into MongoDB
        contact_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "message": message
        }
        contacts_collection.insert_one(contact_data)

        # Set success message
        success_message = "Thank you for contacting us! We'll get back to you soon."

    return render_template('contact.html', success_message=success_message)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        predicted_label, confidence_score = process_image(file_path)
        
        return render_template('result.html',
                               image_path=file_path,
                               filename=filename,
                               predicted_label=predicted_label,
                               confidence_score=confidence_score)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)

