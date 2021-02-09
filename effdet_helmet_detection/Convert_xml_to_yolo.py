import xml.etree.ElementTree as ET
import os
import csv
import sys
import cv2
import pandas as pd 
import shutil
import glob
#import collections

def FindDuplicateInstanceID(InstanceIDList):
    testListDict = {}
    returnlist =[]
    
    for item in InstanceIDList:
        print("Item", item)
        try:
            testListDict[item] += 1
        except:
            testListDict[item] = 1
    for a_key, corresponding_value in testListDict.items():
        if corresponding_value > 1:
            returnlist.append([a_key, corresponding_value])
    print(len(returnlist))
    return returnlist



# Get input folder path which contains image and xml files
input_folder_path  = sys.argv[1]




inputFolderName = path = input_folder_path
Taglist = []
InstanceIDList = []
ZeroFilesList = []
DuplicateInstanceIDFilesList = []
# Taglist.append(["FileName", "Class ID", "BBox Num", "Box Dim"])

# train_file = open('custom_data/train.txt','w')
# test_file =open('custom_data/test.txt','w')

cnt = 0
total = 23666
# for path, subdirs, files in os.walk(inputFolderName):

#     if len(files) > 0:
#         for dirFile in files:
#             filename = os.path.join(path,dirFile)

            # print(filename + '\n',end="")
            
#             if filename.endswith(".txt"):
#                 # dest = 'custom_data/labels/' 
#                 # shutil.copy(filename,dest)
#                 # os.remove(filename)
#                 cnt += 1 
#                 # print(int((total//2)*0.20))
#                 # if(cnt < int((total//2)*0.20)):
#                 #     test_file.write(filename + '\n')
#                 # else:
#                 #     train_file.write(filename + '\n')
# print(cnt)
#                 # dest = "custom_data/images/"
#                 # shutil.copy(filename,dest)
#                 # print(filename)
#                 # cnt+=1
#                 # file = filename.split('/')[2].split('.')[0]
#                 # print(path + '/' + file + '.txt')
#                 # if os.path.exists(path + '/' + file + '.txt') == False:
#                 #     print("yes",end=" ")

# print(cnt)

            

            # print(filename)
            # print(filename,"---->",dest)

            
# if filename.endswith(".txt"):
cnt = 0

filename = 'custom_data/Dataset/test/Helmet/0ba4a702f67ab14d.txt'

file = filename.split('/')[2].split('.')[0]

with open(filename, "r") as tf:
                    lines = tf.readlines()
                    # print(lines)
                    writefile_str=""
                    filedir = 'custom_data/Dataset/test/Helmet/0ba4a702f67ab14d.jpg'
                    # print(os.path.exists(filedir))
                    img = cv2.imread(filedir)
                    height, width, depth = img.shape

                    # print(img.shape)
                    for i in range(len(lines)):
                        tag_list = lines[i].split(' ')
                        print(tag_list)
                        if tag_list[0] == "0":

                            # x1,y1,x2,y2 = (float)(tag_list[1]), (float)(tag_list[2]), (float)(tag_list[3]), (float)(tag_list[4])
                            # w = x2 - x1
                            # h = y2 - y1
                            # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                            # print(bbox_val)

                            yolo_x = float(tag_list[1])
                            yolo_y = float(tag_list[2])
                            yolo_w = float(tag_list[3])
                            yolo_h = float(tag_list[4])
                            


                            
                            # yolo_x = float( ((x1 + (w/2)) / width) )
                            # yolo_y = float( ((y1 + (h/2)) / height) )
                            # yolo_w = float(w / width)
                            # yolo_h = float(h / height)
                            yolobox = [yolo_x,yolo_y,yolo_w,yolo_h]
                            x1, y1 = int((yolo_x + yolo_w/2)*width) , int((yolo_y + yolo_h/2)*height)
                            x2, y2 = int((yolo_x - yolo_w/2)*width), int((yolo_y - yolo_h/2)*height)
                            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,128,0),2)




                            cv2.imshow('img',img)
                            cv2.imwrite('custom_data/gt_iou/'+ '0ba4a702f67ab14d.jpg' ,img)
                            # cnt +=1 
                            # if(cnt > 100):
                            #     break
                            # cv2.waitKey(1)
                            
            #                 #print(arg3,arg4)
            #                 writefile_str += "0 "+ str(yolo_x) + " " + str(yolo_y) + " "+ str(yolo_w) + " " + str(yolo_h)+ "\n"

            #     new_file_name = open(filename, "w")
            #     new_file_name.write(writefile_str)
                






            # if filename.endswith(".xml"):

            #     outputFilename = dirFile.strip('.xml')
            #     img = cv2.imread(os.path.join(path,outputFilename),1)
            #     # print(img.shape)
            #     img_width = float(img.shape[1])
            #     img_height = float(img.shape[0])


            #     txt_name = outputFilename.strip('.jpg') + '.txt'
            #     # print(outputFilename)
            #     # print(txt_name)
            #     # print(path)
            #     yolo_file_path = path + '/' + txt_name 
            #     # print(yolo_file_path)

            #     yolo_file = open(yolo_file_path,'w')
            #     tree = ET.parse(filename)
            #     root = tree.getroot()
            #     for project in root.findall('project'):
            #         name = project.get('name')
            #         InstanceIDList = []
            #     #print name
            #         for clas in project.findall('class'):
            #             class_id = clas.get('id')
            #             for box in clas.findall('boundingbox') or clas.findall('Polyline'):

            #                 x1 = int(box.find('x1').text)
            #                 y1 = int(box.find('y1').text)
            #                 x2 = int(box.find('x2').text)
            #                 y2 = int(box.find('y2').text)
            #                 w = int(box.find('w').text)
            #                 h = int(box.find('h').text)
            #                 boxdim = [x1,y1,x2,y2]
            #                 # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            #                 def sorting(l1, l2):
            #                     if l1 > l2:
            #                         lmax, lmin = l1, l2
            #                         return lmax, lmin
            #                     else:
            #                         lmax, lmin = l2, l1
            #                         return lmax, lmin


            #                 # Yolo Format Converted
            #                 yolo_x = float( ((x1 + (w/2)) / img_width) )
            #                 yolo_y = float( ((y1 + (h/2)) / img_height) )
            #                 yolo_w = float(w / img_width)
            #                 yolo_h = float(h / img_height)
            #                 yolobox = [yolo_x,yolo_y,yolo_w,yolo_h]


            #                 x1, y1 = int((yolo_x + yolo_w/2)*img_width) , int((yolo_y + yolo_h/2)*img_height)
            #                 x2, y2 = int((yolo_x - yolo_w/2)*img_width), int((yolo_y - yolo_h/2)*img_height)

            #                 revboxdim = [x1,y1,x2,y2]
            #                 # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)

            #                 boxID = box.get('id')

            #                 Taglist.append([filename,class_id,boxID,boxdim])
            #                 yolo_string = "0 "
            #                 yolo_string += ''.join(str(e) + ' ' for e in yolobox) + '\n' 
            #                 # print(yolo_string)
            #                 yolo_file.write(yolo_string)



                        # cv2.imshow('img',img)
                        # cv2.waitKey(1)


