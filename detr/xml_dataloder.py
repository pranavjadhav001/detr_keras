import os
import sys
import cv2
import random
import numpy as np
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import math

class XML_Loader(tf.keras.utils.Sequence):
    def __init__(self, base_dir, batch_size, shuffle=True, augment=False, resize_dim=None):
        self.shuffle = shuffle
        self.augment = augment
        self.resize_dim = resize_dim
        self.batch_size = batch_size
        self.base_dir = base_dir
        #base_dir = 'D:/DATASET/FaceMask/train/'
        images_path = glob.glob(os.path.join(self.base_dir, '*.jpg'))
        self.image_xml_path = [(imgpath, imgpath.replace('.jpg', '.xml')) for imgpath in images_path]
        images_path = None
        print('image_xml_path:',len(self.image_xml_path),' | batch_size:',self.batch_size)
        self.class_map = {'background':0,'face':1,'face_mask':2}
        self.indexes = np.arange(len(self.image_xml_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)
  
        
    def letterbox_image_label(self, img,bboxes,desired_size=(416,416)):
        h,w,_ = img.shape
        new_h,new_w = desired_size
        h_ratio,w_ratio = new_h/h,new_w/w
        scale = min(h_ratio,w_ratio)
        new_img = cv2.resize(img,(int(scale*w),int(scale*h)))
        pad_img = np.pad(new_img,((math.floor((new_h-new_img.shape[0])/2), math.ceil((new_h-new_img.shape[0])/2)),\
                                 (math.floor((new_w-new_img.shape[1])/2), math.ceil((new_w-new_img.shape[1])/2)),(0, 0)))
        bboxes[...,0] = bboxes[...,0]*img.shape[1]
        bboxes[...,1] = bboxes[...,1]*img.shape[0]
        bboxes[...,2] = bboxes[...,2]*img.shape[1]
        bboxes[...,3] = bboxes[...,3]*img.shape[0]
        new_bboxes = bboxes.copy()
        new_bboxes[...,0] = bboxes[...,0] - bboxes[...,2]/2
        new_bboxes[...,1] = bboxes[...,1] - bboxes[...,3]/2
        new_bboxes[...,2] = bboxes[...,0] + bboxes[...,2]/2
        new_bboxes[...,3] = bboxes[...,1] + bboxes[...,3]/2
        new_bboxes[...,0:4] = new_bboxes[...,0:4]*scale
        new_bboxes[...,0] = new_bboxes[...,0]+math.floor((new_w-new_img.shape[1])/2)
        new_bboxes[...,1] = new_bboxes[...,1]+math.floor((new_h-new_img.shape[0])/2)
        new_bboxes[...,2] = new_bboxes[...,2]+math.floor((new_w-new_img.shape[1])/2)
        new_bboxes[...,3] = new_bboxes[...,3]+math.floor((new_h-new_img.shape[0])/2)
        new_bboxes = new_bboxes.astype(np.float32)
        bboxes = new_bboxes.copy().astype(np.float32)
        bboxes[...,0] = ((new_bboxes[...,0] + new_bboxes[...,2])//2)/desired_size[0]
        bboxes[...,1] = ((new_bboxes[...,1] + new_bboxes[...,3])//2)/desired_size[0]
        bboxes[...,2] = (new_bboxes[...,2] - new_bboxes[...,0])/desired_size[0]
        bboxes[...,3] = (new_bboxes[...,3] - new_bboxes[...,1])/desired_size[0]
        return pad_img, bboxes

    
    def __len__(self):
        # returns the number of batches
        return len(self.image_xml_path) // self.batch_size

    def preprocess_image(self,img):
        return preprocess_input(img)
    
    def __getitem__(self, index):
        # returns one batch
        # Generate indexes of the batch
        local_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        imgbatch = []
        targetbatch = []
        
        for Lindex in local_indexes:
            try:
                imgpath, xmlpath = self.image_xml_path[Lindex]
                img = cv2.imread(imgpath)
                if img is None:
                    continue
                    
                ih,iw,_ = img.shape
                xmltree = ET.parse(xmlpath).getroot()
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                target = []
                for obj in xmltree.iter('object'):
                    difficult = int(obj.find('difficult').text) == 1
                    name = obj.find('name').text.lower().strip().lower()
                    bbox = obj.find('bndbox')
                    # get face rect
                    bndbox = [int(bbox.find(pt).text) for pt in pts]
                    target.append([int((bndbox[2]+bndbox[0])/2), int((bndbox[1]+bndbox[3])/2),\
                                   int((bndbox[2]-bndbox[0])), int((bndbox[3]-bndbox[1])), self.class_map[name]])
                target = np.array(target, dtype=np.float32)
                target[:,0] = target[:,0]/iw
                target[:,1] = target[:,1]/ih
                target[:,2] = target[:,2]/iw
                target[:,3] = target[:,3]/ih
                img, target[:,:4] = self.letterbox_image_label(img,target[:,:4],desired_size=self.resize_dim)
                img = self.preprocess_image(img)
                targetbatch.append(target)
                imgbatch.append(img)
            except:
                pass
        imgbatch = np.array(imgbatch, dtype=np.float32)
        return imgbatch, np.array(targetbatch)

    def on_epoch_end(self):
        # option method to run some logic at the end of each epoch: e.g. reshuffling
        self.indexes = np.arange(len(self.image_xml_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)
      

    