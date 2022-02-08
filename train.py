#from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
#from configparser import ConfigParser
#from generator import AugmentedImageSequence
from sys import base_exec_prefix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Dense


def main():

    # DIRS
    root = './experiment/'
    experiment_path = os.path.join(root, '1')

    output_weights_path = os.path.join(experiment_path, 'best_weights.h5')

    CSV_PATH = './data_split/'
    train_path = os.path.join(CSV_PATH, 'train.csv')
    val_path = os.path.join(CSV_PATH, 'val.csv')
    test_path = os.path.join(CSV_PATH, 'test.csv')

    batch_size = 32  # total samples / batch_size / 10)
    steps =  # total samples / batch_size / 5)

    train_pd = pd.read_csv(train_path)
    val_pd = pd.read_csv(val_path)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    # TODO checkpoint
    checkpoint = ModelCheckpoint(
        output_weights_path,
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )

    callbacks = [
        checkpoint,
        #TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1,
                          verbose=1, mode="min", min_lr=1e-8),
        # auroc,
    ]

    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessin_function=tf.keras.applications.densenet.preprocess_input)
    val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    label_names = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

    train_generator = train_data_gen.flow_from_dataframe(
        dataframe=train_pd,
        directory="./images",
        x_col="Image Index",
        y_col=label_names,
        batch_size=32,
        shuffle=True,
        class_mode="raw",
        target_size=(224, 224),
        )

    val_generator = val_data_gen.flow_from_dataframe(
        dataframe=val_pd,
        directory="./images",
        x_col="Image Index",
        y_col=label_names,
        batch_size=32,
        shuffle=False,
        class_mode="raw",
        target_size=(224, 224))

    img_input = tf.keras.Input(shape=(224, 224, 3))

    optimizer = Adam(lr=0.001)

    auc = tf.keras.metrics.AUC(multi_label=True)

    base_exec_prefix = tf.keras.applications.DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=(224, 224, 3),
        weights='imagenet,
        pooling="avg")

    x = base.output
    predictions = Dense(14, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=img_input, outputs=predictions)

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer, metrics=[auc, 'accuracy'])


if __name__ == "__main__":
    main()
