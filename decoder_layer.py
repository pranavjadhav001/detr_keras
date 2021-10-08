import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DecoderLayer(layers.Layer):
    def __init__(self,num_heads,key_dim,feature_dim,ff_dim,dropout):
        super().__init__()
        self.multiheadatt1 = layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,\
            dropout=dropout)
        self.multiheadatt2 = layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,\
            dropout=dropout)

        self.feed_forward_layer = keras.Sequential([
            layers.Dense(ff_dim,activation='relu'),\
            layers.Dense(feature_dim)])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

    def call(self,input,query_pos,pos_embeddings,encoder_outputs,training,\
        look_ahead_mask=None,padding_mask=None):
        query = key = input + query_pos
        attention_output1 = self.multiheadatt1(query=query,key=key,value=input,\
            attention_mask=look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1,training=training)
        out1 = self.layernorm1(input+attention_output1)
        query = out1+query_pos
        key = encoder_outputs+pos_embeddings
        attention_output2 = self.multiheadatt2(value=encoder_outputs,key=key,\
                                               query=query,attention_mask=padding_mask)
        attention_output2 = self.dropout1(attention_output2,training=training)
        out2 = self.layernorm2(out1+attention_output2)

        ffn_output = self.feed_forward_layer(out2)
        ffn_output = self.dropout2(ffn_output,training=training)
        out2 = self.layernorm3(out1+ffn_output)
        return out2