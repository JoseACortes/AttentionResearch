import tensorflow as tf

class LongShortTermMemoryNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_embedding = tf.keras.layers.Dense(128, activation='softmax')
        self.lstm = tf.keras.layers.LSTM(128, return_state=True)
        self.densea = tf.keras.layers.Dense(64, activation='relu')
        self.denseb = tf.keras.layers.Dense(64, activation='softmax')
        self.readout = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        lastfeatures = inputs[:, :, -1]
        embedding = self.initial_embedding(lastfeatures)
        whole_seq_output, final_memory_state, final_carry_state = self.lstm(inputs, initial_state=[embedding, embedding])
        dense = self.densea(final_carry_state)
        dense = self.denseb(dense)
        return self.readout(dense)