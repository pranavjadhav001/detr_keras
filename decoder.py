import tensorflow as tf
from decoder_layer import DecoderLayer
from utils import positional_encoding
from tensorflow.keras import layers

class Decoder(layers.Layer):
    def __init__(self,num_decoder,num_heads,key_dim,feature_dim,ff_dim,dropout):
        super(Decoder, self).__init__()
        patches = 100
        self.num_decoder = num_decoder
        #self.pos_encoding = positional_encoding(patches, feature_dim)
        self.dec_layers = [DecoderLayer(num_heads,key_dim,feature_dim,ff_dim,dropout)
                           for _ in range(num_decoder)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs,query_pos,pos_embeddings,encoder_outputs,training,
           look_ahead_mask, padding_mask):
        x = inputs
        for i in range(self.num_decoder):
            x = self.dec_layers[i](input=x,query_pos=query_pos,pos_embeddings=pos_embeddings,\
                encoder_outputs=encoder_outputs,training=training,look_ahead_mask=look_ahead_mask,\
                padding_mask=padding_mask)
        return x