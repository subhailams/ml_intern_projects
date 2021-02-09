#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import pandas as pd 
from skimage import io
from skimage import color
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
# import seaborn as sns
# from dask.array.image import imread
# from dask import bag, threaded
# from dask.diagnostics import ProgressBar
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")



import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
#Use the generated model 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


# In[4]:


from tensorflow.keras.models import model_from_json

tags = { "C0": "safe",
"C1": "txt-R",
"C2": "talk-R",
"C3": "txt-L",
"C4": "talk-L",
"C5": "radio",
"C6": "drink",
"C7": "reach",
"D" : "Distracted"}

with open('mobilenetv1_29mar_8class_moreaug.json','r') as f:
    model = model_from_json(f.read())

model.load_weights('models/mobilenetv1_29mar_8class_moreaug.h5')

sgd = optimizers.SGD(lr = 0.005) # try 0.01 - didn't converge and 0.005 , 0.001 best acc of 11%

model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()



# In[ ]:


# image_array = []
cnt = 0
src = 'distracted_subha_new.mp4'
cap = cv2.VideoCapture(src)
frameno = 0
frame_count =cap.get(cv2.CAP_PROP_FRAME_COUNT)
# url = "http://56.76.210.13:8080/shot.jpg"
image_array = []
while(frameno < frame_count):

    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # frame = cv2.imdecode(img_arr, -1)

    ret,frame = cap.read()
    frameno += 1
    if ret == True:

        
        frame = frame[100:950,0:1500]
        org = frame
        # print(org.shape)

        frame = cv2.resize(frame,(224,224))
        
        predict_test = np.array(frame).reshape(1,224,224,3).astype('float32')
        prediction = model.predict(predict_test)
        prediction = list(prediction[0])
        safe_pred = round(prediction[0]*100,2)
        dist_pred = round(max(prediction)*100,2)
        predicted_class = 'C'+str(prediction.index(max(prediction)))
        if (predicted_class == 'C0' or predicted_class == 'C9'):
            predicted_class = 'C0'
        else:
            cv2.putText(org , "Distracted " , (900,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2 )

        cv2.putText(org, tags[predicted_class] , (900,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2 )
        image_array.append(org)
    else:
        break
        # cv2.imshow('frame',org)
        # cv2.waitKey(1)
        # if cnt > 1000:
        #   end_time = time.time()
        #   break

out = cv2.VideoWriter('Predicted_distracted_subha_new.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, (1500,850))

for i in range(len(image_array)):
    out.write(image_array[i])
out.release()



# In[ ]:


# test = []

# for img in test_image:
#     test.append(img)




# predict_test = np.array(test).reshape(-1,224,224,3).astype('float32')
# prediction = model.predict(predict_test)
# print(prediction[0])

# print(len(test_image))
# # labels is the image array
# i = 0
# fig, ax = plt.subplots(20, 1, figsize = (500,500))

# # for i in range(len(test_image)):
# # for i in range(len(test_image)):
# for i in range(len(test_image)):
#     safe_pred = round(prediction[i][0]*100,2)
#     dist_pred = round(max(prediction[i])*100,2)
#     if ( safe_pred >= dist_pred and prediction[i][9] == max(prediction[i])*100,2 ):
#         predicted_class = 'C0'
#     else:
#         cv2.putText(clean_image[i], "Distracted: ", (800,900), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2 )
#         predicted_class = 'D'
#     print("safe: ", safe_pred)
#     print("Dist: ",dist_pred)
# #     predicted_class = 'C'+str(np.where(prediction[i] == max(prediction[i]))[0][0])
# #     print(predicted_class)
# #     cv2.putText(clean_image[i],  tags[predicted_class] , (800,700), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2 )
# # #     cv2.putText(clean_image[i], "Safe: "  , (800,800), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2 )
# # #     cv2.putText(clean_image[i], "Distracted: ", (800,900), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2 )
# #     ax[i].set_title(tags[predicted_class])
# #     ax[i].imshow(clean_image[i])
# #     plt.show


# In[ ]:





# In[ ]:


# # video = mpy.VideoFileClip('distracted_sanju_full.mp4')
# # cnt = 0
# # print(video.fps * video.duration)
# # while(1):
# #     frame = video.get_frame(cnt)
# # #     print(frame.shape)
# #     cv2.imwrite('sanju_frames_full_mp/'+ str(cnt) + '.jpg',frame)
# #     cnt += 1
# # #     if(cnt % 10 == 0):
# # #         break
    

# cap = cv2.VideoCapture('distracted_sanju_allclass.mp4')
# print(cap.isOpened())
# cnt = 1
# total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(total)
# print(fps)
# print(total/fps)
# # cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if ret:
# #         cv2.imwrite('sanju_frames_full/'+ str(cnt) + '.jpg',frame)
# #         cnt += 1


# In[4]:


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


# In[56]:


# s

