from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from  tensorflow.keras.losses import Loss


class WeightedBinaryCrossentropy(Loss):

    def __init__(self, class_labels, df, reduction='auto'):
        super().__init__(reduction=reduction)
        self.class_labels = class_labels
        self.df = df
        self.class_weights = {}
        self.positive_weights = {}
        self.negative_weights = {}
        self.N = df.shape[0]
        print("NUmber of samples passed to WeightedBinaryCrossentropy: {}".format(self.N))

        self.calculate_weights()
    

    def calculate_weights(self):
        for label in self.class_labels:
            self.positive_weights[label] = sum(self.df[label] == 0) / self.N 
            self.negative_weights[label] = sum(self.df[label] == 1) / self.N
        print(self.class_labels)
        print(self.positive_weights.keys())
        print(self.negative_weights.keys())
        print([k for k in self.positive_weights.keys()])
        print(np.array_equal([k for k in self.positive_weights.keys()], self.class_labels))

    def weighted_binary_crossentropy(self, y_true, y_hat):
        loss = np.zeros(shape=(y_true.shape[0]))
        # niepotrzebny komentarz
        y_hat = tf.convert_to_tensor(y_hat)
        y_true = tf.cast(y_true, y_hat.dtype)
        for i, key in enumerate(self.class_labels):
            first_term = self.positive_weights[key] * y_true[:,i] * K.log(y_hat[:,i] + K.epsilon())
            second_term =  self.negative_weights[key] * (1 - y_true[:,i]) * K.log(1 - y_hat[:,i] + K.epsilon())
            loss -= (first_term + second_term) #tutaj robimy sume dla wszystkich sampli powino miec ksztalt (14, ); w oficjalnej implementacji tensorflow w binary_crossentropy jest mean a nie suma /zeby bylo tak jak oficjalnie wystarczy podzielic sume przez ilosc labeli
        return loss

    def call(self, y_true, y_hat):
      return self.weighted_binary_crossentropy(y_true, y_hat)        
