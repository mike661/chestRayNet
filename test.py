import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from sklearn import metrics


def auc(true, pred):
    label_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    auc_dict = {}
    for index, label_name in enumerate(label_names):
        fpr, tpr, thresholds = metrics.roc_curve(true[:,index], pred[:,index])
        auc_dict[label_name] = metrics.auc(fpr, tpr)

    for x in auc_dict.keys():
        auc = auc_dict[x]
        print('{} auc: {}'.format(x, auc))
    print('Mean AUC: {}'.format(np.array(auc_dict.values).mean())) 
    return auc_dict   


def main():

    CSV_PATH = './data_split/'
    weights_path = './experiment/45/best_weights-050-0.81.h5'
    test_path = os.path.join(CSV_PATH, 'test.csv')
    batch_size = 32
    test_pd = pd.read_csv(test_path)
    test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    label_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test_pd,
        directory="./images",
        x_col="Image Index",
        y_col=label_names,
        batch_size=batch_size,
        shuffle=False,
        class_mode="raw",
        target_size=(224, 224),
        interpolation='lanczos',
    )

    img_input = tf.keras.Input(shape=(224, 224, 3))

    optimizer = Adam(learning_rate=0.001)

    base = tf.keras.applications.DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=(224, 224, 3),
        weights=None,
        pooling="avg")

    x = base.output
    predictions = Dense(14, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=img_input, outputs=predictions)

    model.load_weights(weights_path)

    # model.compile(loss="binary_crossentropy",
    #               optimizer=optimizer, metrics=[auc, 'accuracy'])

    result = model.predict(test_generator)

    print(result.shape)
    print(test_generator.labels.shape)
    bla = auc(test_generator.labels, result)

if __name__ == "__main__":
    main()

