import numpy as np
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.keras.models import Model, load_model
import natsort
from keras.utils import to_categorical
import sys


from data import DataSet
from tqdm import tqdm

# from extractor import Extractor
# from VGGExtractor import VGGExtractor
from MobFaceExtractor import MobFaceExtractor


# config =  tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
 

def get_class_one_hot(classes,class_str):
		# Encode it first.
		label_encoded = classes.index(class_str)

		# Now one-hot it.
		label_hot = to_categorical(label_encoded, len(classes))

		assert len(label_hot) == len(classes)

		return label_hot


def extract_features(seq_length=50, class_limit=3, image_shape=(320, 240, 3),cls='training'):
    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape,model='mobileface')
    # print(data.get_data())
    classes = data.classes

    # get the model.
    # model = Extractor(image_shape=image_shape)
#     vgg_extract = VGGExtractor('resnet50')


    mobface_extract = MobFaceExtractor()


    mood_dict = {'Alert' : '0', 'Low' : '5' , 'Drowsy' : '10' }

    # X,y = [],[] 
    for video in data.data:
        # print(video)
        if video[2] != '29':
            path = os.path.join('face_images/sequences_mobface_512', 'training' ,  video[1]+ '_' + video[2] + '-' + str(seq_length) + \
                '-features')  # numpy will auto-append .npy
            if video[1] == '29':
                path_frames = os.path.join('face_images/', 'testing', video[1])
            else:
                path_frames = os.path.join('face_images/', 'training', video[1])
            
    #         path = os.path.join('/DATA/DMS/new_data', 'sequences_mobface_512', 'training' ,  video[1]+ '_' + video[2] + '-' + str(seq_length) + \
    #             '-features')  # numpy will auto-append .npy
    #         if video[1] == '29':
    #             path_frames = os.path.join('/DATA/DMS/','new_data', 'testing', video[1])
    #         else:
    #             path_frames = os.path.join('/DATA/DMS/','new_data', 'training', video[1])



            # Get the frames for this video.
            filename = mood_dict[str(video[1])] + '_' + str(video[2])
            frames = glob.glob(os.path.join(path_frames, filename + '*jpg'))
            frames  = natsort.natsorted(frames,reverse=False)
            # print(len(frames))

            # # Now downsample to just the ones we need.

            print(video[2] + ":" + str(len(frames)))
            
        
            # Now loop through and extract features to build the sequence.
            print('Appending sequence of the video:',video)
            sequence = []

            cnt = 0
            for image in frames[1000:10000]:

                if os.path.isfile(path + '_' + str(cnt) + '.npy'):
                    continue

                features = mobface_extract.extract(image)

                cnt+=1
                # print('Appending sequence of image:',image,' of the video:',video)
                sequence.append(features)
                
                if cnt % seq_length == 0 and cnt > 0 and cnt < 15000:
                    np.save(path+str(cnt)+'.npy',sequence)
                    # X.append(sequence)
                    # y.append(get_class_one_hot(classes,video[1]))
                    sequence = []
                if cnt > 11000:
                    break
            # print(np.array(X).shape)
            # print(np.array(y).shape)
            print('Sequences saved successfully', path)
    # return np.array(X),np.array(y)




 # Skipping frames seq
# def extract_features(seq_length=1000, class_limit=3, image_shape=(320, 240, 3)):
#     # Get the dataset.
#     data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)
#     # print(data.get_data())

#     # get the model.
#     model = Extractor(image_shape=image_shape)

#     # Loop through data.
#     pbar = tqdm(total=len(data.data))
#     # print(data.data)
#     # print(len(data.data))
#     for video in data.data:

#     #     # Get the path to the sequence for this video.
#         path = os.path.join('data_all', 'sequences', video[0] ,  video[1]+ '_' + video[2] + '-' + str(seq_length) + \
#             '-features')  # numpy will auto-append .npy

#         # Check if we already have it.
#         if os.path.isfile(path + '.npy'):
#             pbar.update(1)
#             continue

#         # Get the frames for this video.
#         frames = data.get_frames_for_sample(video)

#         # # Now downsample to just the ones we need.
#         # print(len(frames))
#         frames = data.rescale_list(frames, seq_length)
#         print(frames)
#         print(video[2] + ":" + str(len(frames)))
        


#         # # Now loop through and extract features to build the sequence.
#         print('Appending sequence of the video:',video)
#         sequence = []
#         for image in frames:
#             features = model.extract(image)
#             sequence.append(features)

#         # Save the sequence.
#         np.save(path, sequence)
        

#     pbar.update(1)

#     pbar.close()


# seq = 50
# mood = {}
# mood['Alert'] = '0'
# mood['Low'] = '5'
# mood['Drowsy'] = '10'

# import os

# def extractor(image_path):

# 	with open('output_graph.pb', 'rb') as graph_file:
# 		graph_def = tf.compat.v1.GraphDef()
# 		graph_def.ParseFromString(graph_file.read())
# 		tf.import_graph_def(graph_def, name='')

# 	with tf.compat.v1.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
# 	    pooling_tensor = sess.graph.get_tensor_by_name('pool_3:0')
# 	    #tf.compat.v2.io.gfile.GFile()
# 	    image_data = tf.compat.v1.gfile.FastGFile(image_path, 'rb').read()	
# 	    pooling_features = sess.run(pooling_tensor, {'DecodeJpeg/contents:0': image_data})
# 	    pooling_features = pooling_features[0]

# 	return pooling_features

# def extract_features():
# 	with open('data/data_file.csv','r') as f:
# 		reader = csv.reader(f)
# 		for videos in reader:
# 			path = os.path.join('data', 'sequences', videos[0], videos[2] + '-' + str(seq) + '-features')
# 			# print(path)
# 			path_frames = os.path.join('data_all', videos[0], videos[1])
# 			par_name = videos[2].split('_')[1]
            
#             print("Reading " +par_name)
#             frames = []
#             for f in os.listdir(path_frames):

#                 if f.split('_')[0] == mood[str(videos[1])] and f.split('_')[1] == par_name:
#                     frames.append(f) 
# 				# frames = glob.glob(os.path.join(path_frames, filename + '/*jpg'))
# 			frames  = natsort.natsorted(frames,reverse=False)
# 				# print(frames)
				
# 			sequence = []
# 			cnt = 0
			
# 			for image in frames:
# 				img_path = path_frames + '/' + image
# 				with tf.Graph().as_default():
# 					features = extractor(img_path)
# 					cnt+=1
# 					print('Appending sequence of image:',image,' of the video:',videos)
# 					sequence.append(features)

# 				if cnt % seq == 0:
# 					np.save(path+str(cnt)+'.npy',sequence)
# 					sequence = []
			
					
# 			print('Sequences saved successfully')
			

# extract_features()																																																																				
