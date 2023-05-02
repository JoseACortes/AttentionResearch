import tensorflow as tf

class FullyConnectedNeuralNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(200*6, activation='softmax')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.d3 = tf.keras.layers.Dense(128, activation = 'softmax')
        self.d4 = tf.keras.layers.Dense(128, activation='softmax')
        self.d5 = tf.keras.layers.Dense(1)
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.d1(x, training=training)
        x = self.d2(x, training=training)
        x = self.d3(x, training=training)
        x = self.d4(x, training=training)
        return self.d5(x, training=training)