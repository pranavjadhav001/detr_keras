import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from utils import cxcywh_to_xyxy

def Matcher(y_true, y_pred_logits, y_pred_bbox):
    """
        y_true: GT list of len batch with each element is an array of 
                shape (n_gt_objects, 5) ; n_gt_objects are number of
                objects in that image sample and 5 -> (cx,cy,w,h,class_label)
                where cordinates are in [0,1] range
        
        y_pred_logits: model output of shape (batch, num_queries, classes)
        y_pred_bbox: model output of shape (batch, num_queries, 4) in [0,1] range -> cx,cy,w,h
    """
    y_pred_bbox = y_pred_bbox.numpy()
    out_loss = 0
    batch = len(y_true)
    b,num_queries,_ = y_pred_logits.shape
    assert b == batch, 'Batch mismatch!!'
    batch_query_indices = []
    y_pred_logits = tf.math.softmax(y_pred_logits).numpy() 
    for i in range(batch):
        out_cls_loss = -y_pred_logits[i][:,(y_true[i][:,-1]).astype(int)]
        out_cdist = distance.cdist(y_pred_bbox[i], y_true[i][:,:4], 'euclidean')
        out_iou = []
        for j in range(len(y_true[i])):
            giou = tfa.losses.giou_loss(cxcywh_to_xyxy(y_pred_bbox[i]), cxcywh_to_xyxy(y_true[i][j,:4][np.newaxis,:]))
            out_iou.append(giou)
        out_iou = -np.array(out_iou).transpose(1,0)
        comb_loss = out_cls_loss + out_cdist + out_iou
        row_ind, col_ind = linear_sum_assignment(comb_loss)
        batch_query_indices.append((row_ind,col_ind))
    return batch_query_indices

def compute_loss(y_true_logits, y_true_bbox,y_pred_logits, y_pred_bbox):
    cls_loss = tf.losses.sparse_categorical_crossentropy(y_true_logits, y_pred_logits)
    box_loss = tf.losses.mean_absolute_error(y_true_bbox, y_pred_bbox)
    cum_loss = box_loss+cls_loss
    cum_loss = tf.reduce_mean(cum_loss)
    return cum_loss