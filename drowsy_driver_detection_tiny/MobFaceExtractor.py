import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class MobFaceExtractor():
    def __init__(self):
        with tf.Graph().as_default():
            self.sess = tf.compat.v1.Session()
            saver = tf.compat.v1.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
            saver.restore(self.sess, 'models/mfn/m1/mfn.ckpt')
            self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
#             self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("MobileFaceNet/Logits/Conv/Conv2D:0")
            self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    
    def image_prep(self,img):
        # img = cv2.imread(img)
        aligned_face = cv2.resize(img, (112,112))
        aligned_face = aligned_face - 127.5
        aligned_face = aligned_face * 0.0078125
        return aligned_face
    
    def flatten_layer(self,input):
        
        input = np.expand_dims(input, axis=0)
        conv_input = keras.Input(shape=(1, 1, 512), name='img')
        output = layers.Flatten()(conv_input)
        model = keras.Model(conv_input, output)
        flatten_feature = model.predict(input)
        return flatten_feature
    
    def extract(self, img_path):
        img = self.image_prep(img_path)
        images = []
        images.append(img)
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder:False}
        
        embeds = self.sess.run(self.embeddings, feed_dict=feed_dict)
        
        features = embeds[0]
        new_feature = self.flatten_layer(features)
        return new_feature[0]
    

    

      



