# Distraction Classification

The goal of Driver Monitoring System is to identify and flag users falling inattentive at the wheel. To measure the alertness of the driver. This work uses the State Farm Driver Distraction Dataset and classifies the driver activity into 8 classes namely
* c0: safe driving
* c1: texting - right
* c2: talking on the phone - right
* c3: texting - left
* c4: talking on the phone - left
* c5: operating the radio
* c6: drinking
* c7: reaching behind

Sample Demo can be found here: [Link](https://drive.google.com/open?id=1qByft3-Mb4gdva5UWtwRedywI9nq72t0) 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them


Install [anaconda](https://www.anaconda.com/distribution/#download-section) and follow installation [steps](https://docs.anaconda.com/anaconda/install/)


### Environment Setup

A step by step series of examples that tell you how to get a development env running
Note: This project contains both [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) models so provided two environments.



Clone this repository using below command

```
git clone https://[bitbucket_username]@bitbucket.org/dms_stwp/distracted_driver_detection.git
```
```
cd distracted_driver_detection
```

Create Environment and activate
```
conda env create --name dms2_env -f dms_env.yml
```
```
conda activate dms2_env
```
Install pip packages
```
pip install -r requirements.txt
```


## Running the tests
In order to run the tests for a video or image use the below script.
Predicted Video Files will be saved to **outputs/** folder


```
python evaluate_distraction.py [model_name] [video/image] [video/image location]
```

To Replace 
[model_name] to run the specific model (See below table for variants)
Default model would be "mobilenetv2_class8_1o4"
[video/image] - input type video or image
[video/image path] - input video/image path location

Example 
```
python evaluate_distraction.py mobilenetv2_class8_1o4 video inputs/demo_video.mp4  
```
Predicted video can be found in outputs/ folder

Choose model_name from below

| model_name                                    | No. of Parameters(appr.) | Training Accuracy(appr.) | Validation Accuracy (appr.) | Time Per Frame(secs) |
|------------------------------------------|--------------------------|--------------------------|-----------------------------|----------------------|
| vgg16_class10                  | 14.7M                    | 86%                      | 84%                         | 0.24                 |
| mobilenetv1_class10_1      | 3.2M                     | 90%                      | 83%                         | 0.38 
| mobilenetv1_class8_1              | 3.2M                     | 97%                      | 94%                         | 0.38                 |
| mobilenetv1_class8_025    | 0.2M                     | 95%                      | 87%                         | 0.37                 |
| mobilenetv2_class8_035     | 0.4M                     | 96%                      | 91%                         | 0.52                 |
| mobilenetv2_class8_1o4 | 4.4M                     | 98.70%                   | 96.30%                      | 0.54                 |
| Efficient Net Lite0 | 3.3M                     | 98.70%                   | 95.80%                      | 0.25                 |
