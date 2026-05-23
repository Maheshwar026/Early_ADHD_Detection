import pandas as pd
import numpy as np
import os
import collections

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump

from pose_extraction import extract_pose_features

print("🔹 Loading HAR dataset...")
df = pd.read_csv("dataset/training_set.csv")

print("🔹 Activity labels found:", df["label"].unique())

X, y = [], []

print("🔹 Extracting pose features...")

for i, row in df.iterrows():
    image_path = os.path.join("dataset/train", row["filename"])
    features = extract_pose_features(image_path)

    if features is not None:
        X.append(features)
        y.append(row["label"])   # ✅ ACTIVITY LABEL

    if (i + 1) % 500 == 0 or (i + 1) == len(df):
        print(f"✅ Processed {i + 1}/{len(df)} images")

X = np.array(X)
y = np.array(y)

print("🔍 Raw activity distribution:", collections.Counter(y))

# Encode activity labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("🔹 Encoded classes:", encoder.classes_)

print("🔹 Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("🔹 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("🔹 Training MULTICLASS SVM...")
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight="balanced"
)

svm.fit(X_train, y_train)

print("🔹 Evaluating model...")
y_pred = svm.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

dump(svm, "activity_svm_model.pkl")
dump(scaler, "scaler.pkl")
dump(encoder, "label_encoder.pkl")

print("\n✅ Activity SVM training completed")
