import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def draw_xml(imgpath, xmlpath):
    img = cv2.imread(imgpath)
    if img is None:
        return img
    
    target = ET.parse(xmlpath).getroot()
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        # get face rect
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = [int(bbox.find(pt).text) for pt in pts]
        img = cv2.rectangle(img, tuple(bndbox[:2]), tuple(bndbox[2:]), (0,0,255), 2)
        img = cv2.putText(img, name, tuple(bndbox[:2]), 3, 1, (0,255,0), 1)
    return img

def read_jpeg_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def resize(image, min_side=800.0, max_side=1333.0):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cur_min_side = tf.minimum(w, h)
    cur_max_side = tf.maximum(w, h)

    scale = tf.minimum(max_side / cur_max_side,
                       min_side / cur_min_side)
    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image


def build_mask(image):
    return tf.zeros(tf.shape(image)[:2], dtype=tf.bool)

def cxcywh_to_xyxy(array):
    """
        input: array: numpy array of shape (batch,4) | 4 -> cx,cy,w,h
        return:array: numpy array of shape (batch,4) | 4 -> x1,y1,x2,y2
    """
    new_array = np.zeros(array.shape)
    new_array[...,0] = array[...,0] - (0.5*array[...,2])
    new_array[...,1] = array[...,1] - (0.5*array[...,3])
    new_array[...,2] = array[...,0] + (0.5*array[...,2])
    new_array[...,3] = array[...,1] + (0.5*array[...,3])
    return new_array

def unnormalize2image(array,width,height):
    """
        input: array: numpy array of shape (batch,4) | 4 -> cx,cy,w,h
        return:array: numpy array of shape (batch,4) | 4 -> x1,y1,x2,y2
    """
    new_array = np.zeros_like(array)
    new_array[...,[0,2]] = array[...,[0,2]]*width
    new_array[...,[1,3]] = array[...,[1,3]]*height
    return new_array

def absolute2relative(boxes, img_size):
    width, height = img_size
    scale = tf.constant([width, height, width, height], dtype=tf.float32)
    boxes *= scale
    return boxes

def xyxy2xywh(boxes):
    xmin, ymin, xmax, ymax = [boxes[..., i] for i in range(4)]
    return tf.stack([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)

def preprocess_input(image):
    channel_avg = tf.constant([0.485, 0.456, 0.406])
    channel_std = tf.constant([0.229, 0.224, 0.225])
    image = (image / 255.0 - channel_avg) / channel_std
    return image

def preprocess_image(image):
    image = resize(image, min_side=800.0, max_side=1333.0)    
    return image, build_mask(image)