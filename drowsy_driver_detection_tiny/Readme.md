# Drowsiness Classification

The goal of Driver Monitoring System is to identify and flag users falling asleep at the wheel. To measure the alertness of the driver. This work uses the UTARLDD dataset with LSTM network.

View this Report to get details [Link](https://docs.google.com/document/d/1xrpdxH1ZM-GQ3IefjP6N0GobwuQFsHop6CEnfRlIP7s/edit?usp=sharing)

Sample Demo can be found here: [Link](https://drive.google.com/open?id=1YSXzQ0lnO3SafPUiXglpkis9Ca_LcA2S)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them


Install [anaconda](https://www.anaconda.com/distribution/#download-section) and follow installation [steps](https://docs.anaconda.com/anaconda/install/)


### Environment Setup

A step by step series of examples that tell you how to get a development env running
Note: This project contains [Tensorflow](https://www.tensorflow.org/) models



Clone this repository using below command

```
git clone https://[bitbucket_username]@bitbucket.org/dms_stwp/drowsy_driver_detection.git
```
```
cd drowsy_driver_detection
```

Create Environment and activate
```
conda env create --name dms_env -f dms_env.yml
```
```
conda activate dms_env
```

Install pip packages
```
pip install -r requirements.txt
```


## Running the tests
In order to run the tests for a video or image use the below script.
Predicted Video Files will be saved to **outputs/** folder


```
python evaluate_drowsiness.py -d [face_detection_model] -e [feature_extraction_model] -p [input video location]
```

To Replace,

[face_detection_model] - Model to detect face of the driver. Choose from ['mtcnn' , 'ultralight']
[feature_extraction_model] - Model to extract spatial feature of the face. Choose from         ["resnet_lstm_large","resnet_lstm_small","mobilenet_lstm"]
[video path] - input video path location

Example:

```
python evaluate_drowsiness.py -m resnet_lstm_large -p inputs/sample_drowsy.mp4
```

Predicted video can be found in **outputs/** folder

Choose face detection and feature extraction model below.

| Backbone  + LSTM Models  | Parameters(appr.)| Model Size  | Training Accuracy(appr.) | Validation Accuracy(appr.) | Inference time with MTCNN (in secs)         | Inference time with Ultralight (in secs)"  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------|--------------------------------|-----------------------------------------------------|----------------------------------------------------|
|  "Resnet 50 + LSTM - 2048 units (resnet_lstm_large)  |  Resnet 50 - 23.5M  LSTM 2048 - 123.7M | Resnet50 - 98 MB         LSTM 2048 - 531MB | 99%                          | 95%                            | FD - 0.12 sec   FE - 0.04 sec LP - 0.05 sec | FD - 0.06 sec FE - 0.04 sec LP - 0.05 sec" |
|  Resnet 50 + LSTM - 512 units  (resnet_lstm_small)     | Resnet 50 - 23.5M  LSTM 512- 5.7M          | Resnet50 - 98 MB         LSTM 512 - 22MB   | 98%                          | 89%                            | FD - 0.12 sec FE - 0.03 sec LP - 0.03 sec   | FD - 0.06 sec FE - 0.03 sec LP - 0.03 sec" |
|  MobileFacenet + LSTM - 512 units (mobilenet_lstm)                                                                                                                                                                                                                                                                                                                                                                    | MobileFaceNet - 0.99M  LSTM 512 - 2.5M     | MobileFaceNet - 4.0 MB LSTM 512 - 10.1MB   | 95%                          | 62%                            | FD - 0.12 sec FE - 0.03 sec LP - 0.04 sec   | FD - 0.06 sec  FE - 0.03 sec LP - 0.04 sec |
|                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                |                                                |                              |                                |                                                     |                                                    |
   *FD - Face Detection(MTCNN or Ultralight) (single face) *FE - Feature Extraction (Resnet or MobileFaceNet)  (single face) *LP - LSTM Prediction (50 faces) 			