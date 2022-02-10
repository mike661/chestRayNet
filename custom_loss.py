from tensorflow.keras import backend as K

class WeightedBinaryCrossentropy:
    def __init__(self, class_labels, df):
        self.class_labels = class_labels
        self.df = df
        self.class_weights = {}
        self.positive_weights = {}
        self.negative_weights = {}
        self.N = df.shape[0]
        print("NUmber of samples passed to WeightedBinaryCrossentropy: {}".format(self.N))

        calculate_weights()
    

    def calculate_weights(self):
        for label in sorted(self.class_labels):
            self.positive_weights[label] = sum(self.df[label] == 0) / self.N 
            self.negative_weights[label] = sum(self.df[label] == 1) / self.N


    def weighted_binary_crossentropy(self, y_true, y_hat):
        loss = float(0)
        # niepotrzebny komentarz
        for i, key in enumerate(self.positive_weights.keys()):
            first_term = self.positive_weights[key] * y_true[i] * K.log(y_hat[i] + K.epsilon())
            second_term =  self.negative_weights[key] * (1 - y_true[i]) * K.log(1 - y_hat[i] + K.epsilon())
            loss -= (first_term + second_term)
            
        return loss
