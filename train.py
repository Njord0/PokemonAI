import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


from common import create_model, process_path
from common import data_dir, test_dir, class_names
from common import TrainingCallback

def do_training():
    images = list(data_dir.glob('*/*.jpg'))

    image_count = len(images)

    test_count = len(list(test_dir.glob('*.jpg')))

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels='inferred',
        validation_split=0.1,
        subset="both",
        seed=1,
        image_size=(224, 224),
        batch_size=32
    )

    AUTOTUNE = tf.data.AUTOTUNE

    # Préparation des données
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)


    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    # test_ds = list_ds.take(val_size)



    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path)
    val_ds = val_ds.map(process_path)
    # test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    #for image, label in train_ds.take(10):
    #  print("Image shape: ", image.numpy().shape)
    #  print("Label: ", label.numpy())

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(32)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    # test_ds = configure_for_performance(test_ds)

    num_classes = len(class_names)

    model = create_model(num_classes)

    # Choix du modele peut varier mais adam reste le meilleur
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    epochs_count = 60
    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_count,
    callbacks=[TrainingCallback()]
    )

    model.save("exort_model.h5")

    for image_path in test_dir.glob('*.jpg'):
        img, label = process_path(str(image_path))

        img_array = tf.expand_dims(img, 0)

        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0])

        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_index]

        confidence_score = 100 * np.max(probabilities)

        print(f"Image: {image_path}, Predicted class: {predicted_class_name}, Confidence score: {confidence_score} ")

    # test_loss, test_accuracy = model.evaluate(test_ds)
    # print("Test accuracy:", test_accuracy)
    # print("Test loss : ", test_loss)

if __name__ == '__main__':
    do_training()






