#from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
#from configparser import ConfigParser
#from generator import AugmentedImageSequence
from sys import base_exec_prefix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os

from tensorflow.keras.layers import Dense

def main():

    #DIRS
    root = './experiment/'
    experiment_dir = os.path.join(root, '1')

    output_weights_path = os.path.join(experiment_dir, 'best_weights.h5')

    CSV_PATH = './data_split/'
    train_path = os.path.join(CSV_PATH, 'train.csv')
    val_path = os.path.join(CSV_PATH, 'val.csv')
    test_path = os.path.join(CSV_PATH, 'test.csv')


    batch_size = 32
    steps = 



    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)


    #TODO checkpoint
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
            #auroc,
        ]

    img_input = tf.keras.Input(shape=(224,224,3))

    optimizer = Adam(lr=0.001)

    auc = tf.keras.metrics.AUC(multi_label=True)

    base_exec_prefix = tf.keras.applications.DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=(224,224,3),
        weights='imagenet,
        pooling="avg")
        
    x = base.output
    predictions = Dense(14, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=img_input, outputs=predictions)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.AUC(), 'accuracy'])

if __name__ == "__main__":
    main()