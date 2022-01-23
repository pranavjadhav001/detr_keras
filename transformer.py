import backbone
import decoder
import encoder
import utils
from utils import positional_encoding,FixedEmbedding
import tensorflow as tf
import numpy as np

class Transformer(tf.keras.Model):
    def __init__(self,num_encoder,num_decoder,num_heads,key_dim,feature_dim,ff_dim,dropout):
        super(Transformer,self).__init__()
        self.encoder = encoder.Encoder(num_encoder,num_heads,key_dim,feature_dim,ff_dim,dropout)
        self.decoder = decoder.Decoder(num_decoder,num_heads,key_dim,feature_dim,ff_dim,dropout)
    
    def call(self,inputs,query_pos,pos_embeddings,padding_mask,training=True):
        b,w,h,dim = inputs.shape
        encoder_output = self.encoder(tf.reshape(inputs,(b,w*h,dim)),pos_embeddings,padding_mask,training)
        y = tf.zeros([b,query_pos.shape[0],query_pos.shape[1]])
        decoder_output = self.decoder(inputs=y,query_pos=query_pos,pos_embeddings=pos_embeddings,encoder_outputs=encoder_output,\
                                      training=False,padding_mask=None,look_ahead_mask=None)
        return decoder_output