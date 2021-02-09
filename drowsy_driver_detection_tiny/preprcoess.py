import os 
# import imutils
# import cv2
import shutil
import pandas as pd
from pandas import read_csv
from subprocess import call
from os import listdir
import numpy as np
# from data import DataSet
from moviepy.editor import *
import glob,os.path
import tensorflow as tf
import glob
from VGGExtractor import VGGExtractor

# from mtcnn.mtcnn import MTCNN

# detector = MTCNN()
# initial = 141 # 7000 frames trained

# data = DataSet(
#         seq_length=seq,
#         class_limit=2,
#         image_shape=(320, 240, 3),
#         initial=initial
#     )



# def makdir():
# 	csv = [f for f in os.listdir('face_csv/report')]
# 	path = 'data_all/'
# 	for i in csv:
# 		par = i.split('_')[0]
# 		mood = i.split('_')[1].split('.')[0]
# 		# print(par)
# 		# print(mood)
# 		if mood == '0':
			
# 			# os.mkdir(di)
# 		if mood == '5':
# 			di = path + '5/' + par
# 			# os.mkdir(di)
# 		if mood == '10':
# 			di = path + '10/' + par
			# os.mkdir(di)	



# Rename
# path = 'Input_100/10/'
# alert = [f for f in os.listdir(path)]
# print(alert)

# for par in alert: 
# # 		# src = path + str(par) + '/' + pic
# 	src = path + str(par)
# 	dest = path + 'Drowsy_' + str(par) 
# 	os.rename(src, dest)



# def resize(p):
# 	path = 'data_all/'+ str(p) + '/'
# 	alert = [f for f in os.walk(path)]
# 	# del alert[7]
# 	print(alert)
# 	for par in alert:
		
# 		pics = [f for f in os.listdir(path + str(par))]
# 		for pic in pics: 
			
		# print("resizing" + pic)
		# 	# src = path + str(par) + '/' + pic
		# img = cv2.imread(path + '/' + pic, cv2.IMREAD_UNCHANGED)
		# 	#print(img.shape)
		# dim = (240, 320)
		# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		# 	# cv2.imshow("img",resized)
		# cv2.imwrite(path +  '/' + pic, resized)

		# img = cv2.imread(path + str(par) + '/' + pic, cv2.IMREAD_UNCHANGED)
		# 	#print(img.shape)
		# dim = (240, 320)
		# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		# 	# cv2.imshow("img",resized)
		# cv2.imwrite(path + str(par) + '/' + pic, resized)
			
		

# resize('0')
# resize('5')
# resize('10')

# def frame_100(p):
# 	path = 'Input/'+ str(p) + '/'
# 	new_path = 'Input_100/'+ str(p) + '/'
# 	alert = [f for f in os.listdir(path)]
# 	del alert[7]
# 	print(alert)
# 	for par in alert:
# 		print("Reading" + str(par))
# 		pics = [f for f in os.listdir(path + str(par))]
# 		for pic in pics: 
# 			num = int(pic.split('_')[2].split('.')[0])
# 			if num  < 100:
# 				# print(pic)
# 			# src = path + str(par) + '/' + pic
# 				img = cv2.imread(path + str(par) + '/' + pic, cv2.IMREAD_UNCHANGED)
	
# 				cv2.imwrite(new_path + str(par) + '/' + pic, img)
			

# frame_100(10)

# def chek():
# 	path = 'Input_100/'+ str(10) + '/'
# 	alert = [f for f in os.listdir(path)]
# 	# del alert[7]
# 	print(alert)
# 	for par in alert:
# 		print("Reading" + str(par))
# 		pics = [f for f in os.listdir(path + str(par))]
# 		# for pic in pics:
# 		print( str(par)+ ": " + str(len(pics)))

# chek()/



# def group():
#     source = 'data_all/training/Drowsy'
#     destination = 'data_all/testing_new/Drowsy'
#     for files in os.listdir(source):
#         if files.split('_')[0] == '10' and  files.split('_')[1] == '50':
#             # print(files)
#             shutil.copy(os.path.join(source,files),destination)
            
		  

# group()

# source = 'data/testing/Low'
# for files in os.listdir(source):
# 	os.remove(os.path.join(source,files))


# file = 'face_csv/csv/49_10.csv'
# df = pd.read_csv(file)
# df['mood'] = 10
# print(df)

# df.to_csv('face_csv/csv/49_10.csv',index=False)

# l=[]
# directory = 'data_all/10/'
# files = list_files(directory, "jpg")
# for f in files:
# 	img = cv2.imread(directory + f, cv2.IMREAD_UNCHANGED)
# 	dim = (240, 320)
# 	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# 			# cv2.imshow("img",resized)
# 	cv2.imwrite(directory +  f, resized)

# 	l.append(resized.shape)
# print(set(l))
# src = '/DATA/DMS/UTARLDD/Fold1_part1/02/0.mov'
# dest = 'ffmeg' + '02-%04d.jpg'
# call(["ffmpeg", "-i", src, dest])


# path = 'data_all/10/'

# dest_dir = 'data_all/testing/10/'
# files = list_files(path, "jpg")

# for f in files:
# 	par = str(f.split('_')[1])
   
# 	if par == '31' or par=='60' or par=='40':
# 		shutil.copy(path + f, dest_dir)


# face_points_to_keep = []
# face_points_to_keep += [9]                     # Nose
# face_points_to_keep += [37,38,39,40,41,42]     # Left Eye
# face_points_to_keep += [43,44,45,46,47,48]     # Right Eye
# face_points_to_keep += [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59] # Outer Lip

# columns_to_keep = ['participant', 'mood', 'time'] + ['px_{x}' for x in face_points_to_keep] + ['py_{x}' for x in face_points_to_keep] +['face_x','face_y','face_w','face_h']
# def get_table(participant, mood, resample_interval='100ms'):
    
#     # # Find File
#     # if base_path is None:
#     #     base = os.path.join('face_csv','csv')
#     # else:
#     #     base = base_path
    
#     files =  'face_csv/csv/' + participant + '_' + mood +'.csv'
    
#     # if(len(files) !=1 ):
#     #     logging.error("Looked for {participant}_{mood}.csv and found {len(files)} tables. Need to match with one table only.")
#     #     raise RuntimeError

#     # Load
#     table = pd.DataFrame()
#     try:
#     	table = pd.read_csv(files)
#     except:
#     	return table

#     # Resample time
#     table['date'] = pd.to_datetime(table.time, unit='s')
#     if resample_interval is not None:
#         table = table.resample(resample_interval, on = 'date').mean()
#     else:
#         table.set_index('date', inplace = True)
    
#     # Drop columns we don't need
#     table = table.filter(columns_to_keep)

#     # # Trim head and tail of the video
#     # table.drop(table[ table['time'] > stop_time ].index, inplace=True)
#     # table.drop(table[ table['time'] < start_time ].index, inplace=True)

#     # Fill missing data
#     table.replace(-1, np.NaN, inplace=True)
#     table.interpolate(inplace=True, limit_direction='both')

#     # Fix Data Types
#     table[['participant', 'mood']] = table[['participant', 'mood']].astype('int32')
#     # pxy_cols = [x for x in table.columns if re.compile('p[xy]_*').match(x)]
#     # table[pxy_cols] = table[pxy_cols].astype('int32')
#     table= table.dropna()

#     return table.shape


# mo = {}

# mo['0'] = 'Alert'
# mo['5'] = 'Low'
# mo['10']='Drowsy'


# alert = ['27', '17', '45', '34', '37', '18', '26', '08', '49', '51']
# low = ['03', '37', '28', '10', '08', '50', '45', '23', '34', '18', '51', '17']
# drowsy = ['50', '03', '45', '51', '05', '27', '18', '28', '26', '49', '10', '46', '37', '34'] 

# al = {}

# def makdir():
# 	l=[]
# 	csv = [f for f in os.listdir('data_all/testing/10')]
# 	df = pd.read_csv('data/data_file.csv')
# 	# # print(df)
# 	# for i in df['type']: 
# 	#     print(i) 
# 	#     print()
# 	# # 	co=1
# 	cnt = 0
# 	for k in ['31', '60' , '40']:
# 		cnt=0
# 		for i in csv:
		
# 			mood = i.split('_')[0]
# 			par = i.split('_')[1].split('.')[0]
# 			if par == k:
# 				cnt+=1
# 		al[k] = cnt
# 	print(al)



	
		# l.append(par)
		# print(par +':' + mood +':' + str(co) )
	# print(set(l))
	# for i in set(l):
		# print("training,Alert," + i)

# makdir()
# cnt = 0
# k=[]
# l=[]

# tot = 5 
# seq = 3
# for i in range(0,tot-seq+1):
# 	l=[]
# 	for j in range(i,i+seq):

# 		l.append(j)

# 	k.append(l)

# f=[]
# for i in k:
# 	f.append(len(i))
# 	# if len(i)!=100:
# 	# 	k.remove(i)
# print(set(f))
# print(len(k))

# path = '/DATA/DMS/UTARLDD/Fold3_part2/32/10_1_edited.mp4'
# clip1 = VideoFileClip(path)
# clip2 = VideoFileClip("/DATA/DMS/UTARLDD/Fold3_part2/32/10_2_edited.mp4")
# final_clip = concatenate_videoclips([clip1,clip2])
# final_clip.write_videofile("/DATA/DMS/UTARLDD/Fold3_part2/32/10.mp4")

# base = '/DATA/DMS/UTARLDD/'

# for fold in range(1,6):
    
#     part1 = base + 'Fold' + str(fold) +'_part1'
#     filesDepth3 = glob.glob(os.path.join(part1,'*'))
#     dirsDepth3 = filter(lambda f: os.path.isdir(f), filesDepth3)
#     print(part1)
#     print([x for x in dirsDepth3])
#     part2 = base + 'Fold' + str(fold) +'_part2'
#     filesDepth2 = glob.glob(os.path.join(part2,'*'))
#     dirsDepth2 = filter(lambda f: os.path.isdir(f), filesDepth3)
   
#     # extract_part(part1, '1')

#     print(part2)
#     print([x for x in dirsDepth2])

#     # extract_part(part2, '2')


# files = [f for f in os.listdir('/DATA/DMS/Output_crop/')]
# cnt =0
# for f in files:
#     if f.split('_')[0] == '0' and f.split('_')[1] == '06':
#         cnt = cnt + 1
#         print(f)
# print(cnt)


# from multiprocessing.dummy import Pool as ThreadPool 
# import skvideo.io
# import imutils
# from skimage import io
# import cv2

# header = ['participant', 'mood', 'fps', 
#           'width', 'height', 
#           'frame_no', 'time', 'face_x','face_y','face_w','face_h']



# def meta_data(path):

#     # cap = cv2.VideoCapture(path)
#     # cap =  skvideo.io.vreader(path)
#     # fps = cap.get(cv2.CAP_PROP_FPS)
#     # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # duration = frame_count/fps

#     probe = skvideo.io.ffprobe(path)
#     fps = probe['video']['@r_frame_rate']
#     fps = int(fps.split('/')[0]) / int(fps.split('/')[1])
#     frames_count = int(probe['video']['@nb_frames'])
#     duration = frames_count/fps
#     width = int(probe['video']['@width'])
#     height = int(probe['video']['@height'])


#     meta = {}
#     meta['fps'] = fps
#     meta['frames_count'] = frames_count
#     minutes = int(duration/60)
#     seconds = duration%60
#     meta['duration'] = str(minutes) + ':' + str(seconds)
#     meta['min'] = int(minutes)
#     meta['width'] = width
#     meta['height'] = height
    

#     return meta

# def capture(cap, par, mood, meta, path):

# 	frameCounter = 0 #iterate through generator object python
# 	frame_len = 1/meta['fps']

# 	result_np = np.empty(shape=(meta['frames_count'] , len(header)))
# 	dropped_frames = []
# 	flag = 0




# 	# fps = FPS().start()
# 	for frame in cap:	    
# 	    #Use MTCNN to detect faces
	   
# 	    # img_path = "/DATA/DMS/Output_crop/" + str(mood) + "_" + str(par) + "_" + str(frameCounter)+ ".jpg"
# 	    # if(os.path.exists(img_path)):
# 	    # 	flag = 1
# 	    # frameCounter = frameCounter + 1
# 	    # 	fps.update()
# 	    # 	continue
	    
# 	    # if(frameCounter == 10):
# 	    # 		break
# 	    frame_row = []
# 	    frame_row.append(par)
# 	    frame_row.append(mood)
# 	    frame_row.append(meta['fps'])
# 	    frame_row.append(meta['width'])
# 	    frame_row.append(meta['height'])
# 	    frame_row.append(meta['frames_count'])
# 	    frame_row.append(frameCounter*frame_len)


# 	    result = detector.detect_faces(frame)

# 	    if result != []:
# 	    	person = result[0]
# 	    	bounding_box = person['box']
# 	    	keypoints = person['keypoints']
# 	    	# print(bounding_box)
# 	    	x1, y1, width, height = person['box']
# 	    	x2, y2 = x1 + width, y1 + height
# 	    	frame_row += [x1, y1, width, height]
# 	    	result_np[frameCounter] = frame_row
	    	
# 	    	if '-1' in frame_row:
# 	    		# print(frameCounter)
# 	    		dropped_frames.append(frameCounter)
# 	    		frameCounter = frameCounter + 1
# 	    		fps.update()
# 	    		continue
# 	    	# pyplot.axis('off')

# 	    	roi_color = frame[y1:y2, x1:x2]
# 	    	croppedImg = roi_color
# 	    	dim = (240, 320)
	    	
# 	    	if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=0):
# 	    		# print(frameCounter)
# 	    		# print(croppedImg.shape)
# 	    		dropped_frames.append(frameCounter)
# 	    		# print(dropped_frames)
# 	    		frameCounter = frameCounter + 1
# 	    		fps.update()
# 	    		continue
	    		
# 	    	croppedImg = cv2.resize(croppedImg, dim, interpolation = cv2.INTER_AREA)
# 	    	# print(croppedImg.shape)
# 	    	# cv2.imshow('img',frame)
# 	    	# cv2.imshow('img1',croppedImg)
# 	    	data_name = "/DATA/DMS/Output_crop/" + str(mood) + "_" + str(par) + "_" + str(frameCounter)+ ".jpg"
# 	    	# cv2.imwrite(data_name, croppedImg)
# 	    	io.imsave(data_name,croppedImg)
# 	    	if cv2.waitKey(1) &0xFF == ord('q'):
# 	    		break
	    	
# 	    	# print(frameCounter)
# 	    	# print(frame_row)

# 	    	# frameCounter = frameCounter + 1
# 	    	# if(frameCounter ==10):
# 	    		break
# 	    	# fps.update()

# 	print("Dropped: ")
# 	print(dropped_frames)   	
# 	# fps.stop()

	
# 	# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# 	# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
# 	if flag == 1:
# 		print("Already exists")
# 	result_df = pd.DataFrame(columns=header, data=result_np)
# 	print(result_df)
# 	result_df.to_csv(os.path.join(csv_output_path, f"{par}_{mood}.csv"))

# 	with open(os.path.join(report_output_path, f"{par}_{mood}.txt"),"w") as report:
#             report.write(f"video: {path}\n")
#             report.write(f"metadata: {meta['fps']} fps, {meta['width']}x{meta['height']}\n")
#             report.write(f"dropped: {dropped_frames}\n")
#             # report.write("elasped time: {:.2f}\n".format(fps.elapsed()))
#             # report.write("approx. FPS: {:.2f}\n".format(fps.fps()))
#             report.write("-"*20)           
	           
    
# def run_participant(path):
#     print("processing", path)
#     p_no = (os.path.basename(path))
#     videos = [f for f in os.listdir(path)]
#     print(videos)  
#     for video in videos:
#       vpath = path + '/' + video
#       vcap = skvideo.io.vreader(vpath)
#       meta = meta_data(vpath)
#       mood = video.split('.')[0]
#       # print(participant + '/' + mood)
#       print("Capturing: " + p_no + '/' + mood)
#       # print(meta)
#       capture(vcap,p_no,mood,meta,vpath)

#     # print(p_no)
#     # p = participent(p_no, path)
#     # p.process_all_moods(width, progress_report)
#     return p_no

# base = '/DATA/DMS/UTARLDD/'

# for fold in range(2,6):
    
#     part1 = base + 'Fold' + str(fold) +'_part1'
#     filesDepth3 = glob.glob(os.path.join(part1,'*'))
#     dirsDepth3 = filter(lambda f: os.path.isdir(f), filesDepth3)
#     print("start pooling")
#     pool = ThreadPool(4)
#     print("Started")
#     results = pool.map(run_participant, dirsDepth3)
#     print(results)

#     pool.close()
#     pool.join() 



# def list_files(directory, extension):
#     return (f for f in listdir(directory) if f.endswith('.' + extension))


# # path = 'data_all/sequences_50/testing/'

# # dest_dir = 'data_all/sequences_50/training/'
# # files = list_files(path, "npy")
# # # print(files)
# # for f in files:
# # 	# print(f)

# # 	if '_60' in f:
# # 		# print(f)

# # 		shutil.copy(path + f, dest_dir)


# path = 'data_all/training/Alert'

# files = os.listdir(path)

# cnt =0
# for f in files:
# 	if f.split('_')[0] == '0' and f.split('_')[1] == '27':
# 		cnt +=1
# print(cnt)



columns_to_keep = ['participant', 'mood', 'fps', 'frame_no', 'time'] + ['face_x','face_y','face_w','face_h']


def get_table(participant, mood):
    

    files =  'Output_crop/csv/' + participant + '_' + mood +'.csv'
    

    # Load
    table = pd.DataFrame()
    try:
    	table = pd.read_csv(files)
    except:
    	return table

    # print(table)

    # Resample time
    table['date'] = pd.to_datetime(table.time, unit='s')
    # if resample_interval is not None:
    #     table = table.resample(resample_interval, on = 'date').mean()
    # else:
    #     table.set_index('date', inplace = True)
    
    # Drop columns we don't need
    table = table.filter(columns_to_keep)

    num = table._get_numeric_data()
    num[num<0] = np.NaN
    num = num.dropna()
    
    

    # Fill missing data
    # table.replace(-1, np.NaN, inplace=True)
    # table.interpolate(inplace=True, limit_direction='both')

    # Fix Data Types
    # table[['participant', 'mood']] = table[['participant', 'mood']].astype('int32')
    
    num['frame_no'] = len(num)
    # pxy_cols = [x for x in table.columns if re.compile('p[xy]_*').match(x)]
    # table[pxy_cols] = table[pxy_cols].astype('int32')

    print(len(num))
    print(files)

    num.to_csv(files)


    return table

# Using Asleep CSV data

def face_from_csv(cap, mood):

	frameCounter = 0
	f=0
	shape = mood.shape[0]
	
	print(shape)
	# mood['mood'] = mood['mood'].replace(101,10)
	if(shape == 0):
		return 1
	
	for frame in cap:	    
	    
	    # Face Coordinates
	    x, y, w, h = int(mood['face_x'].iloc[f]), int(mood['face_y'].iloc[f]), int(mood['face_w'].iloc[f]), int(mood['face_h'].iloc[f])
	    # print(x,y,w,h)
	  
	    frame = imutils.resize(frame, width=360)

	    roi_color = frame[y:y+h, x:x+w]
	    croppedImg = roi_color
	    dim = (240, 320)
	    # cv2.resize(croppedImg, dim, interpolation = cv2.INTER_AREA)
	    # croppedImg = imutils.resize(croppedImg, width = 240)
	    # croppedImg = imutils.resize(croppedImg, height = 320)
	    if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=0):
	        frameCounter = frameCounter + 1
	        continue
	    # print(croppedImg.shape)
	    moo = mood['mood'][f]
	    par = mood['participant'][f]
	    if  par < 10:
	    	par = '0' + str(par)
	
	    # cv2.imshow('img',frame)
	    # cv2.imshow('img1',croppedImg)
	    data_name = "data_all/"  +  str(moo) +'/'+ str(moo) + '_' + str(par) + '_' + str(frameCounter)+".jpg"
	    mood_fol = "data_all/"  + str(moo) +'/'+ str(par) + '/' + str(moo) + '_' + str(par) + '_' + str(frameCounter)+".jpg"
	    # print(data_name)
	    # print(mood_fol)print(f)
	    # print(croppedImg.shape)
	    
	    # # cv2.imwrite(data_name, croppedImg)
	    skimage.io.imsave(data_name,croppedImg)
	    skimage.io.imsave(mood_fol,croppedImg)
	    
	    if cv2.waitKey(1) &0xFF == ord('q'):
	       break
	    frameCounter = frameCounter + 1
	    f=f+1
	    if(f == shape):
	    	break

# df_test = df_test.append({'drowsy': ,'alert':,'Video': filenames_test[k], 'Accuracy': accuracy_test, 'Prediction': j[0],'Prediciton_class': 'Alert'},ignore_index = True)

# df = pd.DataFrame()


# csv_files = os.listdir('Output_crop/csv')
# report_files = os.listdir('Output_crop/report')

# dropped = [1161]

# to_drop = get_table('53', '0')
# print(str(len(to_drop) - len(dropped)))


# print(os.path.exists('/DATA/DMS/Output_crop/10_06_136.jpg'))



# for f in csv_files:
# 	par = f.split('_')[0]
# 	mood = f.split('_')[1].split('.')[0]
# 	table = get_table(par, mood)
# 	# print(f + " - " + str(len(table) == table['frame_no'][0]))

# 	if (len(table) < 15000):
# 		print(f + " - " + str(len(table)))



# mood = get_table('27', '0')
# print(mood)


# example of creating a face embedding
from tensorflow import keras
from tensorflow.python.keras import backend as k
# from keras_vggface.vggface import  VGGFace
from keras.preprocessing.image import load_img, img_to_array
from keras_vggface import utils
from tensorflow.keras.layers import Input

from MobFaceExtractor import  MobFaceExtractor
import cv2


# create a vggface2 model
# resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
#                                 pooling='avg') 
# summarize input and output shape
# print('Inputs: %s' % resnet50_features.inputs)
# print('Outputs: %s' % resnet50_features.outputs)


path_frames = 'Output_crop/All/'





# def image2x(image_path):
#         img = load_img(image_path, target_size=(224, 224))
#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = utils.preprocess_input(x, version=1)  # or version=2
#         return x

# frames = glob.glob(os.path.join(path_frames,'*jpg'))
# fvecs = None
# img_path = frames[0]
# image = image2x(img_path)
# print(image.shape)
# fvecs= resnet50_features.predict(image)
# # print(resnet50_features.predict(image))
# print(fvecs.shape)

# np.save(path_frames + "eg.npy",fvecs)

# image_shape = (320,240,3)
# model = VGGExtractor()


# fvecs2 = model.extract(img_path)
# print(fvecs2)
# print(fvecs2.shape)
# def load_model():

#     saver = tf.compat.v1.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
#     saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

# sess = tf.Session()
# model = load_model()

# with tf.Graph().as_default():
# 	sess = tf.compat.v1.Session()
# 	saver = tf.compat.v1.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
# 	saver.restore(sess, 'models/mfn/m1/mfn.ckpt')
# model = MobFaceExtractor()
# for img_path in frames:
# 	fvecs = model.extract(img_path)
# 	print(fvecs)
# print(fvecs.shape)





# np.save(path_frames + "inc.npy",fvecs2)


            # features = model.extract(image)
        #     cnt+=1
        #     # print('Appending sequence of image:',image,' of the video:',video)
        #     sequence.append(features)
            
        #     if cnt % seq_length == 0:
        #         np.save(path+str(cnt)+'.npy',sequence)
        #         sequence = []
        #     if cnt > 15000:
        #         print(cnt)
        #         break

        # print('Sequences saved successfully')
from moviepy.editor import VideoFileClip

def get_video_metadata(video_path):
	c = VideoFileClip(video_path)
	rotation = c.rotation
	fps = c.fps
	c.close()
	return rotation,fps

print(get_video_metadata('demo.webm'))



		    
		