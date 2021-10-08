from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def build_backbone(height=256,width=256,d_model=256):
    base_model = ResNet50(weights='imagenet', include_top=False,input_shape=(height,width,3))
    x = base_model.output
    x = layers.Conv2D(filters=d_model,kernel_size=1, use_bias=False)(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = True
    return model
