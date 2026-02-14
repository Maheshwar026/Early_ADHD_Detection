import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load training CSV
df = pd.read_csv("dataset/training_set.csv")

# ADHD risk actions
adhd_risk_actions = [
    "running", "dancing", "clapping", "cycling", "fighting"
]

# Map labels
df["adhd_risk"] = df["label"].apply(
    lambda x: "ADHD_Risk" if x in adhd_risk_actions else "Normal"
)

print("Class distribution:")
print(df["adhd_risk"].value_counts())

# Image parameters
IMG_SIZE = (128, 128)   # reduced size to save memory
BATCH_SIZE = 32

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="dataset/train",
    x_col="filename",
    y_col="adhd_risk",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="dataset/train",
    x_col="filename",
    y_col="adhd_risk",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

print("Train batches:", train_generator.samples)
print("Validation batches:", val_generator.samples)
