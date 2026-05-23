from joblib import load
from pose_extraction import extract_pose_features

svm = load("activity_svm_model.pkl")
scaler = load("scaler.pkl")
encoder = load("label_encoder.pkl")

high_motor_activities = ["running", "cycling", "jumping", "dancing"]

def predict_image(image_path):
    features = extract_pose_features(image_path)

    if features is None:
        return {
            "result": "Pose not detected ❌",
            "confidence": 0
        }

    features = scaler.transform([features])

    probs = svm.predict_proba(features)[0]
    class_id = probs.argmax()

    # ✅ CONVERT numpy float → python float
    confidence = float(round(probs[class_id] * 100, 2))

    activity = encoder.inverse_transform([class_id])[0]

    if activity.lower() in high_motor_activities:
        risk = "High Motor Activity (Possible ADHD Risk)"
    else:
        risk = "Low Motor Activity"

    result_text = (
        f"Detected Activity: {activity}<br>"
        f"ADHD Risk Indicator: {risk}"
    )

    return {
        "result": result_text,
        "confidence": confidence
    }