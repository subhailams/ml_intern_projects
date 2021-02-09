
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)
#tf.config.gpu.set_per_process_memory_growth(True)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam, RMSprop
from extract_features import extract_features
import json
from data import DataSet
import time
import os.path
import numpy as np
import sys
import pandas as pd
import csv
import datetime
 

seq=50
frame = 8000
test_frame = 2000
initial = 10
test_initial = 20

# extract_features(
#     seq_length=seq,
#     class_limit=2,
#     image_shape=(320, 240, 3),cls='training')

# X,y = extract_features(
#     seq_length=seq,
#     class_limit=2,
#     image_shape=(320, 240, 3),cls='training')

# print(X.shape)
# print(y.shape)

# X_val,y_val = extract_features(
#     seq_length=seq,
#     class_limit=2,
#     image_shape=(320, 240, 3),cls='testing')


# print(X_val.shape)
# print(y_val.shape)
# df = pd.read_csv("new_data/data_file.csv", header = None)

# test_no = df[df.iloc[:,0] == 'testing'].shape[0]
# train_no = df[df.iloc[:,0] == 'training'].shape[0]

# test_no = df[df.iloc[:,0] == 'testing'].shape[0]

data = DataSet(
        seq_length=seq,
        class_limit=2,
        image_shape=(320, 240, 3),
#         initial=initial
    )



X, y, paths_train = data.get_all_sequences_in_memory('training', frame, seq,initial)
print("X.shape", X.shape)
print("y.shape", y.shape)

X_val, y_val, paths_val = data.get_all_sequences_in_memory('testing',test_frame, seq, test_initial)
print("X_val.shape", X_val.shape)
print("Y_val.shape", y_val.shape)

# X_test, y_test,paths_test = data.get_all_sequences_in_memory('validation',test_frame, seq, test_initial)
# print("X_test.shape" ,X_test.shape)
# print("y_test.shape" ,y_test.shape)


print(data.get_classes())

features = 512
batch_size = 4
nb_epoch = 10
rm = ResearchModels(len(data.classes),'lstm',data.seq_length,features)
print(rm.model.summary())

log_dir="logs/model_mobface_512_new/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpointer = ModelCheckpoint(filepath='models/model_mobface_512_new.hdf5', verbose=1, save_best_only=True)

rm.model.fit(X,y,
         batch_size=batch_size,
         validation_data=(X_val, y_val),
         verbose=1,
         callbacks=[tensorboard_callback,checkpointer],
 		epochs=nb_epoch)
model_json = rm.model.to_json()
with open("models/model_mobface_512_new.json",'w') as json_file:
 	json_file.write(model_json)

rm.model.save_weights("models/model_mobface_512_new.h5")
print("Model saved")
    





# Testing Results

# with open('models/model_vggface_class2_apr6.json','r') as f:
#     model = model_from_json(f.read())

# model.load_weights('models/model_vggface_class2_apr6.h5')
    

# metrics = ['accuracy']
# optimizer = Adam(lr=0.00005)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
# print(model.summary())

# predictions_test = model.predict(X_test)
# loss_test, accuracy_test = model.evaluate(X_test, y_test)

# print("Testing Loss: " + str(loss_test))
# print("Testing Accuracy: " + str(accuracy_test))
    


       


# python prediction.py --train frames count --test frames count -- initial postion for testing
# initial position for testing = train frme count / seq  

# Grid creation in face

# import numpy as np
# import cv2 
# import sys


# # Load the image
# img = cv2.imread("test.jpg")

# # Grid lines at these intervals (in pixels)
# # dx and dy can be different
# dx, dy = 30,30

# # Custom (rgb) grid color
# grid_color = [255,255,255]

# # Modify the image to include the grid
# img[:,::dy,:] = grid_color
# img[::dx,:,:] = grid_color

# # Show the result
# cv2.imshow('img',img)
# cv2.waitKey(1000)


