
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop,SGD
from collections import deque
import sys

class ResearchModels():
	def __init__(self, nb_classes, model, seq_length,features_length):

		# Set defaults.
		self.seq_length = seq_length
		self.load_model = load_model
# 		self.saved_model = saved_model
		self.nb_classes = nb_classes
		self.feature_queue = deque()

		# Set the metrics. Only use top k if there's a need.
		metrics = ['accuracy']
		#if self.nb_classes >= 10:
		#   metrics.append('top_k_categorical_accuracy')

		# Get the appropriate model.
		if model == 'lstm':
			print("Loading LSTM model.")
			self.input_shape = (seq_length, features_length)
			self.model = self.lstm()
			
		# Now compile the network.
		optimizer = Adam(lr=0.00005)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
						   metrics=metrics)
	
		print(self.model.summary())

	def lstm(self):
		"""Build a simple LSTM network. We pass the extracted features from
		our CNN to this model predomenently."""
		# Model.
		#print(self.input_shape)

		# 2048D feature map (VGGFace)
# 		model = Sequential()
# 		model.add(LSTM(512, return_sequences=True, input_shape=self.input_shape))
#         model.add(Flatten())
# 		model.add(Dense(512,activation='softmax'))
# 		model.add(Dropout(0.5))
# 		model.add(Dense(256,activation='softmax'))
		
# 		model.add(Dense(self.nb_classes, activation='softmax'))

		model = Sequential()
		
		model.add(LSTM(512, return_sequences=True, input_shape=self.input_shape,dropout=0.7))
		
		model.add(Dense(512,activation='softmax'))
		model.add(Dropout(0.5))
		model.add(Dense(256,activation='softmax'))
		model.add(Flatten())
				
		
		model.add(Dense(self.nb_classes, activation='softmax'))


		# 192D feature map (MobFace)
#         model = Sequential()
#         model.add(LSTM(192, return_sequences=True, input_shape=self.input_shapedropout=0.2))
#         model.add(LSTM(192, return_sequences=True))
#         model.add(Flatten())
#         model.add(Dense(self.nb_classes, activation='softmax'))
	
		return model

	
