import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
from PIL import Image
    
        
def define_model(model_name):
    models = {
        
    'vgg16_class10' : 'models/model_vgg16_all_layers',
    'mobilenetv1_class10_1' :'models/mobilenet_sgd_nolayers',
    'mobilenetv1_class8_1' : 'models/mobilenetv1_29mar_8class_moreaug',
    'mobilenetv1_class8_025' : 'models/mobilenet_alpha0.25_small',
    'mobilenetv2_class8_035' : 'models/mobilenetv2_alpha35_small',
    'mobilenetv2_class8_1o4' : 'models/mobilenetv2_alpha1.4',
    'mobilenetv3_class8_large' : 'models/mobilenetv3_large',
    'efficientnetlite' : 'models/effnetlite0_new.pt', 
      }
    
    if model_name  == 'efficientnetlite':
        

        def load_checkpoint(filepath):
            model = torch.load(filepath)
            return model

        model = load_checkpoint('models/effnetlite0_new.pt')
        model.cuda()
        return model
    
    if model_name == 'mobilenetv3_class8_large':
        from keras.models import model_from_json,Model
        from keras.applications import MobileNetV3Large
        from keras.layers import GlobalAveragePooling2D, Dense
        from keras import optimizers
            
        base_model=MobileNetV3Large(alpha=1.0,weights='imagenet',include_top=False)

        x=base_model.output
        x=GlobalAveragePooling2D()(x)

        preds=Dense(8,activation='softmax')(x) #final layer with softmax activation
        model = Model(inputs=base_model.input, outputs=preds)
        model.load_weights(models[model_name] + '.hdf5')
        sgd = optimizers.SGD(lr = 0.005)
        model.compile(loss='categorical_crossentropy',optimizer = sgd,metrics=['accuracy'])
        # print(model.summary())
        
        return model
    
    from tensorflow.keras.models import model_from_json    
    with open(models[model_name]+'.json','r') as f:
        
        model = model_from_json(f.read())
        # print(model.summary())
    
    model.load_weights(models[model_name] + '.h5')
        
        


    # Compile CNN model
   
    print("Model Name: ",model_name)
   
    return model