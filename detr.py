import tensorflow as tf
import numpy as np
import backbone
import decoder
import encoder
import utils
from utils import positional_encoding,FixedEmbedding
from transformer import Transformer

class Detr(tf.keras.layers.Layer):
    def __init__(self,num_queries,num_classes=10):
        super(Detr,self).__init__()
        self.backbone_model = backbone.build_backbone()
        self.patch_size = (8,8)
        self.num_queries = num_queries
        self.pos_encoder = positional_encoding(64,256)
        self.query_embed = FixedEmbedding((self.num_queries, 256),name='query_embed')
        self.transformer = Transformer(2,2,2,256,256,512,0.1)
        self.class_embed = tf.keras.layers.Dense(num_classes)
        self.bbox_embed = tf.keras.Sequential([
                        tf.keras.layers.Dense(256),
                        tf.keras.layers.Dense(256),
                        tf.keras.layers.Dense(4,activation='sigmoid')])
    
    def call(self,inputs):
        x = backbone_model(inputs)
        hs = self.transformer(x,self.query_embed(None),self.pos_encoder,None,True)
        bbox_output = self.bbox_embed(hs)
        class_output = self.class_embed(hs)
        return bbox_output,class_output