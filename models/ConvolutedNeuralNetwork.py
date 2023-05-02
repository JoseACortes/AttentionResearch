import tensorflow as tf

class ConvolutedNeuralNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2da = tf.keras.layers.Conv2D(1, (1, 64), activation='relu')
        self.maxpoola = tf.keras.layers.MaxPooling2D((1, 2))
        self.conv2db = tf.keras.layers.Conv2D(1, (3, 16), activation='softmax')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(64, activation='softmax')
        self.d3 = tf.keras.layers.Dense(32, activation='softmax')
        self.readout = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.conv2da(x)
        x = self.maxpoola(x)
        x = self.conv2db(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.readout(x)