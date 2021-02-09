#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pycocotools.coco import COCO
import requests

coco = COCO('datasets/COCO/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['motorcycle'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
print("imgIds: ", len(imgIds))
print("images: ", len(images))

# for im in images:
# #     print("im: ", im)
#     img_data = requests.get(im['coco_url']).content
#     with open('datasets/COCO/images/' + im['file_name'], 'wb') as handler:
#         handler.write(img_data)


# In[8]:


from pycocotools.coco import COCO
import requests

coco = COCO('datasets/COCO/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
print("imgIds: ", len(imgIds))
print("images: ", len(images))

for i in range(0,5000):
    im = images[i]
#     print("im: ", im)
    img_data = requests.get(im['coco_url']).content
    with open('datasets/COCO/images/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)


# In[3]:


with open('annotations_download_' + classes + '.csv', mode='w', newline='') as annot:
for im in images:
annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
for i in range(len(anns)):
    annot_writer = csv.writer(annot)
    #annot_writer.writerow([im['coco_url'], anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2], anns[i]['bbox'][1] + anns[i]['bbox'][3], classes])
    annot_writer.writerow(['downloaded_images/' + im['file_name'], int(round(anns[i]['bbox'][0])), int(round(anns[i]['bbox'][1])), int(round(anns[i]['bbox'][0] + anns[i]['bbox'][2])), int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3])), classes])
    #print("anns: ", im['coco_url'], anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2], anns[i]['bbox'][1] + anns[i]['bbox'][3], 'person')
    annot.close()

