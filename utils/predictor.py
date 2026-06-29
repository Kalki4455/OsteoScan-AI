import tensorflow as tf
from utils.preprocess import preprocess_image

model = tf.keras.models.load_model(
    "models/osteoscan_model.keras",
    compile=False
)
"models/osteoscan_model.keras"


def predict(image):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)[0][0]

    if prediction >= 0.5:

        label = "Osteoporosis"

        confidence = prediction * 100

    else:

        label = "Normal"

        confidence = (1 - prediction) * 100

    return label, confidence