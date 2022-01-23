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

class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_shape, **kwargs):
        super().__init__(**kwargs)
        self.embed_shape = embed_shape

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel', shape=self.embed_shape,
                                 initializer='zeros', trainable=True)

    def call(self, x=None):
        return self.w

def cxcywh_to_xyxy(array):
    """
        input: array: numpy array of shape (batch,4) | 4 -> cx,cy,w,h
        return:array: numpy array of shape (batch,4) | 4 -> x1,y1,x2,y2
    """
    new_array = np.zeros(array.shape)
    new_array[:,0] = array[:,0] - (0.5*array[:,2])
    new_array[:,1] = array[:,1] - (0.5*array[:,3])
    new_array[:,2] = array[:,0] + (0.5*array[:,2])
    new_array[:,3] = array[:,1] + (0.5*array[:,3])
    return new_array

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