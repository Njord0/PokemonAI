import tensorflow as tf
import numpy as np
import keras

from common import process_path
from common import class_names, test_dir

def do_test():
    model = keras.models.load_model('model.h5')

    for image_path in test_dir.glob('*.jpg'):
        img, label = process_path(str(image_path))

        img_array = tf.expand_dims(img, 0)

        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0])

        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_index]

        confidence_score = 100 * np.max(probabilities)

        print(f"Image: {image_path}, Predicted class: {predicted_class_name}, Confidence score: {confidence_score} ")

if __name__ == '__main__':
    do_test()
