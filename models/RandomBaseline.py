import tensorflow as tf

class RandomBaseline(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def call(self, inputs):
        rand = tf.random.normal([tf.shape(inputs)[0], 1])
        return rand