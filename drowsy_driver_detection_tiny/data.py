import csv
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical
from VGGExtractor import VGGExtractor
from MobFaceExtractor import MobFaceExtractor
from natsort import natsorted

class DataSet():
	def __init__(self, seq_length=50, class_limit=3, image_shape=(320, 240, 3),model='resnet50'):
		self.seq_length = seq_length
		self.class_limit = class_limit
		self.sequence_path = os.path.join('face_images', 'sequences_mobface_512/')
		self.max_frames = 40000  # max number of frames a video can have for us to use it
# 		self.i =  initial
		# Get the data.
		self.data = self.get_data()
		self.classes = self.get_classes()
		self.data = self.clean_data()
		self.image_shape = image_shape
		# self.vgg_extract = VGGExtractor(model)
# 		self.mob_extract = MobFaceExtractor()
	
	def get_data(self):
		with open(os.path.join('new_data', 'data_file.csv'), 'r') as fin:
			reader = csv.reader(fin)
			data = list(reader);#random.shuffle(data)
			return data


	def get_classes(self):
		classes = []
		for item in self.data:
			if item[1] not in classes:
				classes.append(item[1])

		# Sort them.
		classes = sorted(classes)
		# print(classes)
		# Return.
		if self.class_limit is not None:
			return classes[:self.class_limit]
		else:
			return classes

	def clean_data(self):
		data_clean = []
		for item in self.data:
			if int(item[3]) >= int(self.seq_length) and int(item[3]) <= self.max_frames and item[1] in self.classes:
				data_clean.append(item)
		return data_clean

	def get_class_one_hot(self, class_str):
		# Encode it first.
		label_encoded = self.classes.index(class_str)

		# Now one-hot it.
		label_hot = to_categorical(label_encoded, len(self.classes))

		assert len(label_hot) == len(self.classes)

		return label_hot
	
	def extract_features(self, train_test):
		mood_dict = {'Alert' : '0', 'Low' : '5' , 'Drowsy' : '10' }
		X,y = [], []
		video_cnt = 0
		data = self.data
		for video in data[0]:
			seq_cnt = 0
			if(video[0] == train_test):
				video_cnt += 1

				path_frames = os.path.join('/DATA/DMS/','new_data', 'training' ,video[1]);print(path_frames);
				filename = mood_dict[str(video[1])] + '_' + str(video[2])
				frames = glob.glob(os.path.join(path_frames, filename + '*jpg'))
				frames  = natsorted(frames,reverse=False)
				print(video[2] + ":" + str(len(frames)))
				print('Appending sequence of the video:',video)
				sequence = []
				cnt = 0
				seq_cnt = 0
				for image in frames:
# 					features = self.vgg_extract.extract(image)
					features = self.mob_extract.extract(image)
					cnt+=1

					sequence.append(features)
					if cnt % self.seq_length == 0 and cnt > 1000 and cnt < 5000:
						X.append(sequence)

						y.append(self.get_class_one_hot(video[1]))
						sequence = []
						break


					if cnt > 18000:
						#print(cnt)
						break
					
			if video_cnt >= 1:
				break
		return np.array(X),np.array(y)

	# Get all sequenes 
	def get_all_sequences_in_memory(self, train_test,hyper,seq,initial):

				#train, test = self.split_train_test()

		print("Loading samples into memory for --> ",train_test)

		X, y, paths = [], [], []
		itr =1
		for videos in self.data:


			if(videos[0] == train_test and int(videos[3])>10000):
				print("Reading : " + str(videos))

				i = initial
				while i <= int(hyper/seq):
					cnt = i*seq
					if cnt>0 and cnt<10000:
						filename = videos[2]
						# print(i)
						sequence, path= self.get_extracted_sequence(videos,cnt,seq,'training')
						
						
						if sequence is None:
							print("Can't find sequence. Did you generate them?")
							raise
						X.append(sequence)
						y.append(self.get_class_one_hot(videos[1]))
						paths.append(path)
					i+=1
					itr+=1
				
				
					
					

		return np.array(X), np.array(y), paths

   # Get all sequenes or single participant 
	def get_all_sequences_in_memory_single(self, train_test,hyper,seq):

				#train, test = self.split_train_test()

		print("Loading samples into memory for --> ",train_test)

		X, y, paths = [], [], []
		
		for videos in self.data:
			# X, y, paths = [], [], []
			itr =1
			# print("Reading:" + str(videos))
			if(videos[0] == train_test and int(videos[3])>10000):
				print("Reading : " + str(videos))

				

				i = self.i
				while itr <= int(hyper/seq):
					cnt = i*seq
					filename = videos[2]
					# print(i)
					# sequence = self.get_extracted_sequence(videos,cnt,seq,train_test)
					# path = os.path.join(self.sequence_path, 'training',videos[1] + '_' + filename + '-' + str(50) + '-' + 'features' + str(cnt)+'.npy')
					# print(path)
					# sequence = np.load(path)
					sequence, path= self.get_extracted_sequence(videos,cnt,seq,'training')
					if sequence is None:
						print("Can't find sequence. Did you generate them?")
						raise

					X.append(sequence)
					y.append(self.get_class_one_hot(videos[1]))
					paths.append(path)
					i+=1
					itr+=1
				
					

		return np.array(X), np.array(y), paths
		
	def split_train_test(self):
		train = []
		test = []
		for item in self.data:
			if item[0] == 'training':
				train.append(item)
			else:
				test.append(item)
		return train, test


	def get_extracted_sequence(self,video,cnt,seq,train_test):
		"""Get the saved extracted features."""
		filename = video[2]
		#print(filename)
		# self.sequence_path = 'data_all/sequences_50'
		path = os.path.join(self.sequence_path, train_test,video[1] + '_' + filename + '-' + str(self.seq_length) + '-' + 'features' + str(cnt)+'.npy')
		
		
		if os.path.isfile(path):
			
			return np.load(path), path
		else:
			print(path)
			return None

	# @staticmethod
	# def get_frames_for_sample(sample):
	# 	mood_dict = {'Alert' : '0', 'Low' : '5' , 'Drowsy' : '10' }
	# 	path = os.path.join('data_all', sample[0], sample[1])
	# 	filename = mood_dict[str(sample[1])] + '_' + str(sample[2])
	# 	# print(filename)
	# 	images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
	# 	# print(len(images))
	# 	return images


	# @staticmethod
	# def rescale_list(input_list, size):

	# 	assert len(input_list) >= size
	# 	skip = len(input_list) // size
	# 	output = [input_list[i] for i in range(0, len(input_list), skip)]
	# 	return output[:size]



	# def get_all_sequences_in_memory(self, train_test, data_type):

	# 	train, test = self.split_train_test()
	# 	data = train if train_test == 'training' else test
	# 	print("Loading %d samples into memory for %s." % (len(data), train_test))

	# 	X, y = [], []
	# 	for row in data:

	# 		if data_type == 'images':
	# 			frames = self.get_frames_for_sample(row)
	# 			frames = self.rescale_list(frames, self.seq_length)
	# 			sequence = self.build_image_sequence(frames)
	# 		else:
	# 			sequence = self.get_extracted_sequence(data_type, row, train_test)

	# 			if sequence is None:
	# 				print("Can't find sequence. Did you generate them?")
	# 				raise
	# 		X.append(sequence)
	# 		y.append(self.get_class_one_hot(row[1]))
	# 	return np.array(X), np.array(y)


	# def get_extracted_sequence(self, data_type, video, train_test):
	# 	"""Get the saved extracted features."""
	# 	filename = video[2]
	# 	path = os.path.join(self.sequence_path, train_test,video[1] + '_' + filename + '-' + str(self.seq_length)+  '-' + 'features.npy')
	# 	# print(path)
	# 	if os.path.isfile(path):
	# 			return np.load(path)
	# 	else:
	# 		print(path)
	# 		# return [1]
	