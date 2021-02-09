#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from skimage import io
from skimage import color


# import tensorflow.keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing import image 
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras import optimizers
# #Use the generated model 
# from tensorflow.keras.models import Model

from keras.models import Model, model_from_json
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.utils import to_categorical



# In[10]:


driver_details = pd.read_csv('../statefarm_dataset/driver_imgs_list.csv',na_values='na')
print(driver_details.head(5))


# In[11]:


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


# In[ ]:


## getting list of driver names

D = []
for features,labels,drivers in train_image:
    D.append(drivers)

## Deduplicating drivers

deduped = []

for i in D:
    if i not in deduped:
        deduped.append(i)
    

## selecting random drivers for the validation set
driv_selected = []
import random
driv_nums = random.sample(range(len(deduped)), 4)
for i in driv_nums:
    driv_selected.append(deduped[i])
print(driv_selected)


# In[12]:


## Randomly shuffling the images

import random
random.shuffle(train_image)

driv_selected = ['p050', 'p015', 'p022', 'p056']


# In[13]:


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


# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications import MobileNetV2
# from keras.applications import MobileNetV3Small
from keras.applications import MobileNetV3Large


base_model=MobileNetV3Large(alpha=1.0,weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer..4
# base_model.summary()

x=base_model.output
x=GlobalAveragePooling2D()(x)

preds=Dense(8,activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)

# from tensorflow.keras.models import model_from_json
# with open('models/mobilenetv2_alpha35_small_3.json','r') as f:
#     model = model_from_json(f.read())

# model.load_weights('models/mobilenetv2_alpha35_small_3.hdf5')

sgd = optimizers.SGD(lr = 0.005) # try 0.01 - didn't converge and 0.005 , 0.001 best acc of 11%

model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:


# Random Eraser
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

def blur(img):
    return (cv2.blur(img,(5,5)))


# In[15]:

from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
# from tensorflow.keras.callbacks import TensorBoard
import json
import datetime

checkpointer = ModelCheckpoint('models/mobilenetv3_large_ckpt.hdf5', verbose=1, save_best_only=True)
# earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# log_dir="logs/mobilenetv3_small/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


datagen = ImageDataGenerator(
    rotation_range=30,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2,
    preprocessing_function=get_random_eraser(),
    brightness_range=[0.2,1.0],
    zoom_range=[0.5,1.0],

    )
     
data_generator = datagen.flow(X_train, y_train, batch_size = 64)

# Fits the model on batches with real-time data augmentation:
mobilenet_model = model.fit_generator(data_generator,steps_per_epoch = len(X_train) / 64, 
                                      callbacks=[checkpointer],
                                      epochs = 40, verbose = 1, validation_data = (X_test, y_test))
	


# model.save_weights("mobilenetv3_large.h5")
# print("Model saved")


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(range(1, len(model.history.history['accuracy']) + 1), model.history.history['accuracy'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Accuracy')
axes[0].plot(range(1, len(model.history.history['val_accuracy']) + 1), model.history.history['val_accuracy'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Accuracy')
axes[0].set_xlabel('Epochs', fontsize = 14)
axes[0].set_ylabel('Accuracy',fontsize = 14)
axes[0].set_title('CNN Dropout Accuracy Trainig VS Testing', fontsize = 14)
axes[0].legend(loc = 'best')
axes[1].plot(range(1, len(model.history.history['loss']) + 1), model.history.history['loss'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Loss')
axes[1].plot(range(1, len(model.history.history['val_loss']) + 1), model.history.history['val_loss'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Loss')
axes[1].set_xlabel('Epochs', fontsize = 14)
axes[1].set_ylabel('Loss',fontsize = 14)
axes[1].set_title('CNN Dropout Loss Trainig VS Testing', fontsize = 14)
axes[1].legend(loc = 'best')

fig.savefig('logs/log_mobilenetv3_large.png')


model_json = model.to_json()
with open("models/mobilenetv3_large_json.json",'w') as json_file:
 	json_file.write(model_json)
    
    
# In[ ]:


# from tensorflow.keras.models import model_from_json

# tags = { "C0": "safe driving",
# "C1": "texting - right",
# "C2": "talking on the phone - right",
# "C3": "texting - left",
# "C4": "talking on the phone - left",
# "C5": "operating the radio",
# "C6": "drinking",
# "C7": "reaching behind",
# "C8": "hair and makeup",
# "C9": "talking to passenger" }


# with open('models/mobilenet_sgd_nolayers.json','r') as f:
#     model = model_from_json(f.read())

# model.load_weights('models/mobilenet_sgd_nolayers.hdf5')

# sgd = optimizers.SGD(lr = 0.001) # try 0.01 - didn't converge and 0.005 , 0.001 best acc of 11%

# model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()






# In[ ]:


# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import cv2
# # labels is the image array
# test_image = []
# i = 0
# fig, ax = plt.subplots(1, 20, figsize = (50,50 ))

# files = os.listdir('../../statefarm_dataset/test')
# nums = np.random.randint(low=1, high=len(files), size=20)
# for i in range(20):
# #     print ('Image number:',i)
#     img = cv2.imread('../../statefarm_dataset/test/'+files[nums[i]])
#     #img = color.rgb2gray(img)
#     img = img[50:,120:-50]
#     img = cv2.resize(img,(224,224))
#     test_image.append(img)
#     ax[i].imshow(img,cmap = 'gray')
#     plt.show

# test = []

# for img in test_image:
#     test.append(img)


# # In[ ]:


# predict_test = np.array(test).reshape(-1,224,224,3).astype('float32')
# prediction = model.predict(predict_test)
# print(prediction[0])


# # labels is the image array
# i = 0
# fig, ax = plt.subplots(20, 1, figsize = (100,100))

# for i in range(20):
#     ax[i].imshow(test[i])
#     predicted_class = 'C'+str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
#     ax[i].set_title(tags[predicted_class])
#     plt.show
    


# # In[ ]:


# from natsort import natsorted
# clean_image = []
# test_image = []

# fig, ax = plt.subplots(1, 20, figsize = (20,20))

# files = natsorted(os.listdir('sanju_frames_allclass'))
# print(len(files))
# nums = np.random.randint(low=1, high=len(files), size=20)
# i = 0
# # j = 1000
# # for i in range(len(files)):
# for i in range(20):
# #     print(nums[i])
#     img = cv2.imread('sanju_frames_allclass/'+files[nums[i]])
#     img = cv2.flip(img, 1)
# #     img = cv2.flip(img, 1)
#     clean_image.append(img)
#     print(img.shape)
# #     print(img.shape)
# #     cv2.rectangle(img,(30,100),(1400,1080),(255,0,0),2)
#     img = img[50:,120:-50]
    
#     img = cv2.resize(img,(224,224))
#     test_image.append(img)
#     ax[i].imshow(img,cmap = 'gray')
#     plt.show


# # test = []

# # for img in test_image:
# #     test.append(img)

# # print(len(clean_image))
# # print(len(test_image))


# # In[ ]:


# get_ipython().system('pip install natsort')

