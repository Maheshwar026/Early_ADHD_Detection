import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("adhd_model.h5")

IMG_SIZE = 128

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        return "ADHD Risk Detected ⚠️"
    else:
        return "Normal Behavior ✅"

# TEST IMAGE (change filename)
test_image_path = "dataset/train/Image_1.jpg"

result = predict_image(test_image_path)
print("Prediction:", result)
