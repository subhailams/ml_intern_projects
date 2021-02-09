import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
from models import ResearchModels
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from extract_features import extract_features
import json
from data import DataSet
import time
import os
import os.path
import numpy as np
import sys
import pandas as pd
import csv
import datetime

seq=50
hyper = 10000
data = DataSet(
        seq_length=seq,
        class_limit=2,
        image_shape=(320, 240, 3),
        model='resnet50'
        )


# with open('models/rm.model_vgg_class2.json','r') as f:
#     model = model_from_json(f.read())
    
# model.load_weights('models/rm.model_vgg_class2.h5')

with open('models/model_vggface_resnet50_512_3.json','r') as f:
    model = model_from_json(f.read())
    
model.load_weights('models/model_vggface_resnet50_512_3.h5')




metrics = ['accuracy']
optimizer = Adam(lr=0.00005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
print(model.summary())
    

new_prediction = []
for video in data.data:
    if video[0] == 'testing':
        print(video)
        X_test = []
        prediction_array = []
        i = 1
        
        while i <= int(hyper/seq):
            X_test = []
            cnt = i*seq
            path = os.path.join(data.sequence_path, 'training',
                   video[1] + '_' + video[2] + '-' + str(seq) + '-' + 'features' + str(cnt)+'.npy')
#             print(os.path.isfile(path))
            if os.path.isfile(path):
                sequence = np.load(path)
                X_test.append(sequence)
                prediction = model.predict(np.array(X_test))[0]
#                 print(prediction)
                if(prediction[0] > prediction[1]):
                    prediction_array.append(0)
                else:
                    prediction_array.append(1)
            i+=1
#         print(prediction_array)
        if len(prediction_array) != 0:
            print("Average: ",sum(prediction_array)/len(prediction_array))
#             prediction_avg_array.append(sum(prediction_array)/len(prediction_array))
            
print("Total Average: ",sum(prediction_avg_array)/len(prediction_avg_array))
                
                
                
            
            
                
            
        
        
       




