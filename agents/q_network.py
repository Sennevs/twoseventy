from tensorflow.keras.layers import Dense, BatchNormalization, Concatenate, Embedding, Flatten, Dropout
from tensorflow.keras import Model


class QNetwork(Model):

    def __init__(self):

        super().__init__()

        self.layer_1 = Concatenate()
        self.layer_2 = Dense(1024, activation='relu', kernel_regularizer=self.kernel_regularizer)
        self.layer_3 = BatchNormalization()
        self.layer_4 = Dense(128, activation='relu', kernel_regularizer=self.kernel_regularizer)
        self.layer_5 = BatchNormalization()
        self.layer_6 = Dense(1, activation='relu', kernel_regularizer=self.kernel_regularizer)

        return

    def call(self, inputs, training=None, mask=None):

        state = inputs[1]
        action = inputs[2]

        y = self.layer_1([state, action])
        y = self.layer_2(y)
        y = self.layer_3(y)
        y = self.layer_4(y)
        y = self.layer_5(y)
        y = self.layer_6(y)

        return y