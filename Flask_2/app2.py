from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json

app = Flask(__name__)

model = load_model('dermnet_densenet121_82.h5')

with open('labels.json') as f:
    label_data = json.load(f)

labels = label_data['labels']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    predicted_description = label_data['descriptions'][predicted_label]
    description = predicted_description['description']
    severity = predicted_description['severity']
    action = predicted_description['action']
    confidence = round(float(predictions[0][predicted_index]) * 100, 2)

    response = {
        "name": predicted_label,
        "confidence": "{:.2f}%".format(confidence),
        "description": description,
        "severity": severity,
        "action": action
    }
    return json.dumps(response, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)
