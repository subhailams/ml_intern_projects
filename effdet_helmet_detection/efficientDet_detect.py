#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git


# In[ ]:


# ! cd Monk_Object_Detection/3_mxrcnn/installation && cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install


# In[ ]:


# !pip install efficientnet-pytorch tensorboardX


# In[7]:


import sys
sys.path.append("Monk_Object_Detection/4_efficientdet/lib/");


# In[8]:


from train_detector import Detector


# In[9]:


# pwd


# In[10]:


gtf = Detector();


# In[11]:


root_dir = ".";
coco_dir = "coco_dataset_3class";
img_dir = ".";
set_dir = "Images";


# In[12]:


gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=512, use_gpu=True)


# In[13]:


gtf.Model();


# In[14]:


gtf.Set_Hyperparams(lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0)


# In[ ]:


gtf.Train(num_epochs=30, model_output_dir="trained/");


# In[ ]:





# In[2]:


# import sys
# sys.path.append("Monk_Object_Detection/4_efficientdet/lib/");
# from src.dataset import CocoDataset
# root_dir = "coco_dataset_3class";
# coco_dir = "coco_dataset_3class";
# img_dir = "";
# set_dir = "Images";
# data = CocoDataset(root_dir, img_dir, set_dir)


# In[6]:


# import cv2
# img = data.load_image(10000)
# # cv2.imwrite("sample.jpg",img)

# # img = cv2.imread('coco_dataset_3class/Images/5692104364f7653c.jpg')
# # import os
# # cv2.imwrite("sample.jpg",img)


# In[ ]:


# print(data.load_classes())

