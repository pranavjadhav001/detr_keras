import tensorflow as tf
from decoder_layer import DecoderLayer
from utils import positional_encoding
from tensorflow.keras import layers

class Decoder(tf.keras.Model):
    def __init__(self,num_decoder,num_heads,key_dim,feature_dim,ff_dim,dropout):
        super(Decoder, self).__init__()
        patches = 100
        self.num_decoder = num_decoder
        self.dec_layers = [DecoderLayer(num_heads,key_dim,feature_dim,ff_dim,dropout)
                           for _ in range(num_decoder)]
        #self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs,query_pos,pos_embeddings,encoder_outputs,training,
           look_ahead_mask, padding_mask):
        x = inputs
        outputs = []
        for layer in self.dec_layers:
            x = layer(input=x,query_pos=query_pos,pos_embeddings=pos_embeddings,\
                encoder_outputs=encoder_outputs,training=training,look_ahead_mask=look_ahead_mask,\
                padding_mask=padding_mask)
            outputs.append(self.norm(x))
        return tf.convert_to_tensor(outputs)