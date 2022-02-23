import argparse
from detr.detr import Detr
import tensorflow as tf
from detr.utils import *
from detr.position_embeddings import PositionEmbeddingSine
import cv2
import matplotlib.pyplot as plt
from detr.image_draw import BboxDraw

parser = argparse.ArgumentParser(formatter_class=
	argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_path',help='path to image',type=str)
args = parser.parse_args()

coco_classes = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]
plotter = BboxDraw('font/FiraMono-Medium.otf', coco_classes)
image = read_jpeg_image(args.image_path)
inp_image, mask = preprocess_image(image)

detr = Detr(num_queries=100,num_classes=len(coco_classes))
detr.build([(1,inp_image.shape[0],inp_image.shape[1],3),\
	(1,inp_image.shape[0],inp_image.shape[1])])
detr.summary()
detr.load_weights('new_detr.h5')


inp_image = preprocess_input(inp_image)
inp_image = tf.expand_dims(inp_image, axis=0)
mask = tf.expand_dims(mask, axis=0)
boxes,logits = detr((inp_image, mask))
probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
scores = tf.reduce_max(probs, axis=-1)
labels = tf.argmax(probs, axis=-1)
boxes = cxcywh_to_xyxy(boxes)

keep = scores[0] > 0.5
labels = labels[0][keep]
scores = scores[0][keep]
boxes = boxes[0][keep]

final_boxes = unnormalize2image(boxes, image.shape[1], image.shape[0])
final_boxes = final_boxes.astype(np.int32)
img = plotter.draw(args.image_path, final_boxes, labels)
plt.imshow(img)
plt.show()