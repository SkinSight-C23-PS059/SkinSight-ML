from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json

app = Flask(__name__)

# Load the trained model
model = load_model('models/model_skinsight.h5')

# Define the labels
labels = [
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Eczema Photos',
    'Nail Fungus and other Nail Disease',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections'
]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = labels[np.argmax(predictions)]
    return json.dumps(predicted_label, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)
