import tensorflow as tf
from encoder_layer import EncoderLayer
from utils import positional_encoding
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self,num_encoder,num_heads,key_dim,feature_dim,ff_dim,dropout):
        super().__init__()
        patches = 256
        self.num_encoder = num_encoder
        self.encoder_layers = [EncoderLayer(num_heads,key_dim,feature_dim,ff_dim,dropout) \
                              for _ in range(num_encoder)]
        self.dropout = layers.Dropout(dropout)
    
    def call(self,inputs,pos_embeddings,padding_mask,training=True):
        x = inputs
        for i in range(self.num_encoder):
            x = self.encoder_layers[i](x,pos_embeddings,\
                training=training,attention_mask=padding_mask)
        return x