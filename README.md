# detr_keras
Tensorflow/keras implementation of End-to-End Object Detection with Transformers(DETR) research paper<br/>
Paper link : https://arxiv.org/abs/2005.12872<br/>

## Implementation
This is an unofficial Tensorflow implementation of DETR,End-to-end Object Detection with Transformers (Carion et al.). This Repo uses tensorflow's official MultiHeadAttention layer for attention. For Backbone, a similiar implementation of Pytorch's Resnet has been used rather than Tensorflow's pretrained resnet architecture due to discrepancies in Model architecture like use of dilation in Pytorch convolution layers, presence bias weights in Tensorflow etc. PositionEmbeddingSine layer has been adopted from official Pytorch's DETR library. Weights have been manually converted to fit the architecture. You can download the converted Pytorch weights in Tensorflow from [here](https://drive.google.com/drive/folders/1dbmJtyRv4tX3oc2r9ucq-B5vN752prrr?usp=sharing).

## Run Detr with Pretrained Coco weights
python3 demo_coco.py --image_path path_to_image

## Results
![alt text](https://github.com/pranavjadhav001/detr_keras/blob/main/images/sample_result.png)

## TO-DO
- [x] Model Architectures
- [x] Loss
- [x] Training
- [ ] Inference
- [x] Pretrained Model
- [x] DataGenerator 
- [ ] Weight conversion scripts

## References
- https://github.com/Leonardo-Blanger/detr_tensorflow/tree/38fc3c586b6767deed09bd7ec6c2a2fd7002346e
- https://github.com/facebookresearch/detr
- https://github.com/qqwweee/keras-yolo3