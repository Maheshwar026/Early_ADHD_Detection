from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def upload():
    return render_template("lab_upload.html")

@app.route("/preview", methods=["POST"])
def preview():
    file = request.files["image"]
    file.save(os.path.join(UPLOAD_FOLDER, "uploaded.jpg"))
    return render_template("lab_preview.html")

@app.route("/predict", methods=["POST"])
def predict():
    output = predict_image(os.path.join(UPLOAD_FOLDER, "uploaded.jpg"))
    return render_template(
        "lab_result.html",
        result=output["result"],
        confidence=output["confidence"]
    )

if __name__ == "__main__":
    app.run(debug=True)