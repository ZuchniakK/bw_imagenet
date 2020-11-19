import tensorflow as tf
import numpy as np
from os.path import join
import os

import sys
import json

from datasets_generator import BWImageNetDataGenerator

DATA_DIRECTORY = '/net/scratch/datasets/AI/imagenet/data'
# DATA_DIRECTORY = 'data'
MODEL_DIRECTORY = 'models'
MAX_TRAIN_EPOCH = 1000
EPOCH_PER_DATASET = 1
VALIDATION_SPLIT = 0.2
TRAIN_N_SAMPLES = 50000
TEST_N_SAMPLES = 50000


def get_bw_model(model_name):
    model_builder = getattr(tf.keras.applications, model_name)

    model = model_builder(
        include_top=True, weights=None, input_tensor=None, input_shape=None,
        pooling=None, classes=1000)

    input_shape = model.layers[0]._batch_input_shape
    print(input_shape)

    x, y = 0, 1
    if input_shape[0] is None:
        x += 1
        y += 1

    new_shape = (input_shape[x], input_shape[y], 1)
    target_size = (input_shape[x], input_shape[y])

    model = model_builder(
        include_top=True, weights=None, input_tensor=None, input_shape=new_shape,
        pooling=None, classes=1000)

    print(model.summary())

    return model, target_size


def train(model_name, batch_size):
    model, target_size = get_bw_model(model_name)
    bw_gen = BWImageNetDataGenerator(directory=DATA_DIRECTORY,
                                     batch_size=batch_size,
                                     target_size=target_size,
                                     validation_split=VALIDATION_SPLIT)


    model_dir = join(MODEL_DIRECTORY, model_name.lower() + '_' + str(batch_size))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    backup = tf.keras.callbacks.experimental.BackupAndRestore(join(model_dir, 'backup'))
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005)
    early_stoping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    csv_logger = tf.keras.callbacks.CSVLogger(join(model_dir, 'train_log.csv'), append=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    model.fit(
        x=bw_gen.train_flow(),
        epochs=MAX_TRAIN_EPOCH,
        callbacks=[backup, lr_reducer, early_stoping, csv_logger],
        validation_data=bw_gen.val_flow(),
        steps_per_epoch=int((TRAIN_N_SAMPLES * (1 - VALIDATION_SPLIT)) / batch_size * EPOCH_PER_DATASET),
        validation_steps=int((TRAIN_N_SAMPLES * VALIDATION_SPLIT) / batch_size * EPOCH_PER_DATASET))

    result = model.evaluate(
        x=bw_gen.test_flow(),
        steps=int(batch_size(TRAIN_N_SAMPLES * VALIDATION_SPLIT)),
        return_dict=True)

    with open(join(model_dir, 'evaluation.txt'), 'w') as file:
        file.write(json.dumps(result))  # use `json.loads` to do the reverse

    model.save(model_dir)


if __name__ == '__main__':
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    train(model_name, batch_size=batch_size)

    # 'ResNet50'