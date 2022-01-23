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

    def call(self,input,pos_embeddings,training,attention_mask):
        query = key = input + pos_embeddings
        attention_output = self.multiheadatt(query = query,key= key ,value = input,\
            attention_mask=attention_mask)
        attention_output = self.dropout1(attention_output,training=training)
        out1 = self.layernorm1(input+attention_output)
        ffn_output = self.feed_forward_layer(out1)
        ffn_output = self.dropout2(ffn_output,training=training)
        out2 = self.layernorm2(out1+ffn_output)
        return out2