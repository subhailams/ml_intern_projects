#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import cv2
import numpy as np
from keras.utils import to_categorical


# In[2]:


driver_details = pd.read_csv('../statefarm_dataset/driver_imgs_list.csv',na_values='na')
print(driver_details.head(5))


# In[3]:



## Getting all the images

train_image = []
image_label = []

num_classes = 8

for i in range(num_classes):
    print('now we are in the folder C',i)
    imgs = os.listdir("../statefarm_dataset/train/c"+str(i))
    print(len(imgs))
    for j in range(len(imgs)):   
        img_name = "../statefarm_dataset/train/c"+str(i)+"/"+imgs[j]
        img = cv2.imread(img_name)
        # img = color.rgb2gray(img)
        img = img[50:,120:-50]
        img = cv2.resize(img,(224,224))
        label = i
        driver = driver_details[driver_details['img'] == imgs[j]]['subject'].values[0]
        train_image.append([img,label,driver])
        image_label.append(i)


# In[4]:


import random
random.shuffle(train_image)

driv_selected = ['p050', 'p015', 'p022', 'p056']


# In[5]:


## Splitting the train and test

X_train= []
y_train = []
X_test = []
y_test = []
D_train = []
D_test = []

for features,labels,drivers in train_image:
    if drivers in driv_selected:
        X_test.append(features)
        y_test.append(labels)
        D_test.append(drivers)
    
    else:
        X_train.append(features)
        y_train.append(labels)
        D_train.append(drivers)
    
print (len(X_train),len(X_test))
print (len(y_train),len(y_test))

## Converting images to nparray. Encoding the Y

X_train = np.array(X_train).reshape(-1,224,224,3)
X_test = np.array(X_test).reshape(-1,224,224,3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print (X_train.shape)


# In[6]:


import tensorflow_hub as hub
from tensorflow.keras import optimizers
module = hub.Module("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2",trainable=True)


# In[7]:


import tensorflow as tf


# In[8]:


feature_extractor_layer = hub.KerasLayer(module,
                                         input_shape=(224,224,3))
feature_extractor_layer.trainable = False


# In[9]:


model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(8,activation='softmax')
])
sgd = optimizers.SGD(lr = 0.005) # try 0.01 - didn't converge and 0.005 , 0.001 best acc of 11%

model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[10]:


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


# In[10]:



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import json
import datetime

# checkpointer = ModelCheckpoint('models/effnetlite0.hdf5', verbose=1, save_best_only=True)
# earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

log_dir="logs/logs_effnetlite0/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=[0.5,1.0],
    height_shift_range=0.1,
    width_shift_range=0.1,
    preprocessing_function=get_random_eraser(v_l=0, v_h=1),
    brightness_range=[0.2,1.0],
    
    )
     
data_generator = datagen.flow(X_train, y_train, batch_size = 64)

# # Fits the model on batches with real-time data augmentation:
mobilenet_model = model.fit_generator(data_generator,steps_per_epoch = len(X_train) / 64, 
                                      callbacks=[tensorboard_callback],
                                      epochs = 200, verbose = 1, validation_data = (X_test, y_test))
	
model_json = model.to_json()
with open("models/effnetlite0.json",'w') as json_file:
 	json_file.write(model_json)

# model.save_weights("models/effnetlite0.h5")
# print("Model saved")

