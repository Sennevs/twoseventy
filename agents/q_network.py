from tensorflow.keras.layers import Dense, BatchNormalization, Concatenate, Embedding, Flatten, Dropout, Input
from tensorflow.keras import Model


class QNetwork(Model):

    def __init__(self):

        super().__init__()

        self.embed_1 = Embedding(18, 18)
        self.flatten_1 = Flatten()


        self.layer_1 = Concatenate()
        self.layer_2 = Dense(512, activation='relu')
        self.layer_3 = BatchNormalization()
        self.layer_4 = Dense(128, activation='relu')
        self.layer_5 = BatchNormalization()
        self.layer_6 = Dense(64, activation='relu')
        self.layer_7 = BatchNormalization()
        self.layer_8 = Dense(1, activation='linear')

        return

    def call(self, inputs, training=None, mask=None):

        state = self.embed_1(inputs[0])
        state = self.flatten_1(state)
        y = self.layer_1([state, inputs[1]])
        y = self.layer_2(y)
        y = self.layer_3(y)
        y = self.layer_4(y)
        #y = self.layer_5(y)
        #y = self.layer_6(y)
        y = self.layer_7(y)
        y = self.layer_8(y)

        return y
