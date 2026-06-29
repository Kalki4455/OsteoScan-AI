import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "datasets/archive"

IMG_SIZE = (224,224)
BATCH_SIZE = 16

model = tf.keras.models.load_model("models/osteoscan_model.keras")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

pred = model.predict(val_data)

pred = (pred > 0.5).astype(int)

print(classification_report(val_data.classes, pred))

print(confusion_matrix(val_data.classes, pred))