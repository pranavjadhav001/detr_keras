import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EncoderLayer(layers.Layer):
    def __init__(self,num_heads,key_dim,feature_dim,ff_dim,dropout):
        super().__init__()
        self.multiheadatt = layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,\
            dropout=dropout)
        self.feed_forward_layer = keras.Sequential([
            layers.Dense(ff_dim,activation='relu'),\
            layers.Dense(feature_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self,input,pos_embeddings=None,training=False,attention_mask=None):
        query = key = input + pos_embeddings
        attention_output = self.multiheadatt(query = query,key= key ,value = input,\
            attention_mask=attention_mask,training=False)
        input += self.dropout1(attention_output,training=training)
        input = self.layernorm1(input)
        ffn_output = self.feed_forward_layer(input)
        input += self.dropout2(ffn_output,training=training)
        input = self.layernorm2(input)
        return input