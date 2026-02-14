from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load trained model
model = load_model("adhd_model.h5")

IMG_SIZE = 128

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    # 🔁 FIXED LOGIC (REVERSED)
    # 0 → ADHD Risk
    # 1 → Normal
    if prediction < 0.5:
        return "ADHD Risk Detected ⚠️"
    else:
        return "Normal Behavior ✅"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    if file.filename == "":
        return "No selected file"

    upload_path = os.path.join("static", "uploaded.jpg")
    file.save(upload_path)

    result = predict_image(upload_path)

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
