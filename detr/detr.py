import tensorflow as tf
import numpy as np
from .backbone import ResNet50Backbone
from .fixed_embedding import FixedEmbedding
from .transformer import Transformer
from .position_embeddings import PositionEmbeddingSine
from tensorflow.keras.layers import Conv2D, ReLU

class Detr(tf.keras.Model):
    def __init__(self,num_queries,num_classes=10):
        super(Detr,self).__init__()
        self.backbone = ResNet50Backbone(name='backbone')
        self.model_dim = 256
        self.transformer = Transformer(6,6,8,32,256,2048,0.0)
        self.input_proj = Conv2D(
            self.model_dim, kernel_size=1, name='input_proj')
        self.patch_size = (25,56)
        self.num_queries = num_queries
        self.pos_encoder = PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True)
        self.query_embed = FixedEmbedding((self.num_queries, 256),name='query_embed')
        self.class_embed = tf.keras.layers.Dense(num_classes+1,name='class')
        self.bbox_embed1 = tf.keras.layers.Dense(256,activation='relu')
        self.bbox_embed2 = tf.keras.layers.Dense(256,activation='relu')
        self.bbox_embed3 = tf.keras.layers.Dense(4,activation='sigmoid')

    def build(self, input_shape=None, **kwargs):
        super(Detr, self).build(input_shape)

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False,
            half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks

    def call(self,inputs):
        x, masks = inputs
        x = self.backbone(x)
        masks = self.downsample_masks(masks, x)
        pos_encoding = self.pos_encoder(masks)
        batch_size,h,w,channels = tf.shape(pos_encoding)[0],tf.shape(pos_encoding)[1],\
        tf.shape(pos_encoding)[2],tf.shape(pos_encoding)[3] 
        pos_encoding = tf.reshape(pos_encoding,[batch_size,h*w,channels])
        x = self.input_proj(x)
        hs = self.transformer(x,self.query_embed(None),pos_encoding,None,False)
        bbox_output = self.bbox_embed1(hs)
        bbox_output = self.bbox_embed2(bbox_output)
        bbox_output = self.bbox_embed3(bbox_output)
        class_output = self.class_embed(hs)
        return bbox_output[-1],class_output[-1]