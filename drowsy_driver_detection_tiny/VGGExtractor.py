from tensorflow import keras
from tensorflow.python.keras import backend as k
from keras_vggface.vggface import  VGGFace
from keras.preprocessing.image import load_img, img_to_array
from keras_vggface import utils
from tensorflow.keras.layers import Input
import numpy as np


class VGGExtractor():
    def __init__(self,model):
        self.model  = model
        
        self.features_extractor = VGGFace(model = self.model , include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')

        
#         self.vgg_features = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3),
#                                 pooling='avg')
#         self.sequeezenet50_features = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3),
#                                 pooling='avg')        
#         self.resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
#                                 pooling='avg')
#         self.sequeezenet50_features = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3),
#                                 pooling='avg')
        
    def image2x(self,image_path):
        # img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(image_path)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        return x

    def extract(self, img_path):
        img = self.image2x(img_path)

        # Get the prediction.
        features = self.features_extractor.predict(img)
        features = features[0]

        return features


