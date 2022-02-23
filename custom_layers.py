import tensorflow as tf


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_shape, **kwargs):
        super().__init__(**kwargs)
        self.embed_shape = embed_shape

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel', shape=self.embed_shape,
                                 initializer='zeros', trainable=True)

    def call(self, x=None):
        return self.w
