import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from common import create_model, process_path
from common import data_dir, class_names
from common import TrainingCallback

def do_training():
    images = list(data_dir.glob('*/*.jpg'))

    image_count = len(images)

    AUTOTUNE = tf.data.AUTOTUNE

    # Préparation des données
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    val_size = int(image_count * 0.3)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path)
    val_ds = val_ds.map(process_path)

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(32)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    num_classes = len(class_names)

    model = create_model(num_classes)

    # Choix du modele peut varier mais adam reste le meilleur
    model.compile(optimizer='sgd',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    epochs_count = 250
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_count,
        callbacks=[TrainingCallback()]
    )

    model.save("model.h5")

if __name__ == '__main__':
    do_training()






