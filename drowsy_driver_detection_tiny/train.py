"""
Train our LSTM on extracted features.
"""

from tensorflow.keras.callbacks import  TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from extract_features import extract_features
import time
import os.path
import sys
import tensorflow as tf



def main():
    """These are the main training settings. Set each before running
    this file."""

    if (len(sys.argv) == 5):
        seq_length = int(sys.argv[1])
        class_limit = int(sys.argv[2])
        image_height = int(sys.argv[3])
        image_width = int(sys.argv[4])
    else:
        print ("Usage: python train.py sequence_length class_limit image_height image_width")
        print ("Example: python train.py 75 2 720 1280")
        exit (1)

    sequences_dir = os.path.join('new_data', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('new_data', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = True # pre-load the sequences into memory
    batch_size = 10
    nb_epoch = 100
    data_type = 'features'
    image_shape = (image_height, image_width, 3)

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)

if __name__ == '__main__':
    main()
