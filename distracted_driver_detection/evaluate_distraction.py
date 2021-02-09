import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys
from load_models import *
import time
import argparse

parser = argparse.ArgumentParser(description='Run inference for driver distraction classification. Eg: python evaluate_distraction.py -m efficientnetlite -t image -p inputs/drink.JPG')

parser.add_argument('-m','--model', default = 'efficientnetlite', help="Name of the model. Select from the list ['vgg16_class10','mobilenetv1_class10_1','mobilenetv1_class8_1','mobilenetv1_class8_025','mobilenetv2_class8_035','mobilenetv2_class8_1o4','mobilenetv3_class8_large','efficientnetlite']")
parser.add_argument('-t','--type', default = 'image', required=True, help='Input type video/image')
parser.add_argument('-p','--path',default = 'inputs/drink.JPG',required=True, help='Location of Input video/image')

args = vars(parser.parse_args())
model_name = args['model']
type = args['type']
path = args['path']


# example: python evaluate_distraction.py -m efficientnetlite -t image -p inputs/drink.JPG

tags = { "C0": "safe driving",
"C1": "texting - right",
"C2": "talking on the phone - right",
"C3": "texting - left",
"C4": "talking on the phone - left",
"C5": "operating the radio",
"C6": "drinking",
"C7": "reaching behind",
"C8": "doing hair/makeup",
"C9": "talking to passenger" }

fps = 0
org_array = []
test_array = []
prediction_array = []
if os.path.exists(path):

    print()
    print("Evaluating {} model on {} -> {}. Prediction results will be saved in folder outputs/".format(model_name,type,path))
    print()

    # Preprocessing input
    print("Please wait! Preprocessing input...",path)


    if type == 'image':
        org_img = cv2.imread(path)
        org_array.append(org_img)
        w,h,c = org_img.shape
        print("Found 1 image")
        img = cv2.resize(org_img,(224,224))
        test_array.append(img)
            


    if type == 'video':
        cap = cv2.VideoCapture(path)
        frame_count = 0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if cap.isOpened():
            while(frame_count < total_frames):
                ret, frame = cap.read()
                if ret:
                    org_img = frame
                    w,h,c = org_img.shape
                    org_array.append(org_img)
                    img = cv2.resize(org_img,(224,224))
                    test_array.append(img)
                    frame_count += 1
            print("Found {} images in {}".format(frame_count,path))
        else:
            print("Please provide valid path")
                    

    print("Done! Preprocessing input")
    print()


    # Loading Model
    print("Loading model...")
    model = define_model(model_name)

    print("Done! Loading model {}".format(model_name))
    print()


    if model_name == 'efficientnetlite':

        def predict(img):
            trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                ])

            image = Image.fromarray((img))
            input = trans(image)
            input = input.view(1, 3, 224,224)
            
            start = time.time()
            scores = model(input.cuda())
            end = time.time()
            # print(model_name,"model -> took {} to predict a single image".format(end-start))
            
            return scores

        start = time.time()
        
        for i in range(0,len(test_array)):
            scores = predict(test_array[i])[0]
            prediction = torch.argmax(scores)
            score_list = scores.tolist()
            pred = prediction.cpu().numpy()
            predicted_class = 'C'+str(pred)

            if prediction == 0:
                cv2.circle(org_array[i], (10,10), 5, (0,128,0), -1)
                cv2.circle(org_array[i], (10,10), 8, (255,255,255), 1)
                cv2.putText(org_array[i], "Attentive" , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 1 )
                # print("Driver Attentive")
            else:
                cv2.circle(org_array[i], (10,10), 5, (0,0,255), -1)
                cv2.circle(org_array[i], (10,10), 8, (255,255,255), 1)
                cv2.putText(org_array[i], "Inattentive - " + str(tags[predicted_class]) , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1 )
                # print("Driver Inattentive -> ",tags[predicted_class])
            prediction_array.append(org_array[i])
        end = time.time()
        fps = len(test_array)/(end - start)
        print("Frames per Second: ",fps)
        if len(test_array) == 1:
            img_name = 'outputs/' + model_name + '_Predicted_image.jpg'
            cv2.imwrite(img_name,prediction_array[0])
            print("Saved results to ",img_name)
        else:
            vid_name = 'outputs/' + model_name + '_Predicted_video.mp4'
            out = cv2.VideoWriter(vid_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, (h,w))
            for i in range(len(prediction_array)):
                out.write(prediction_array[i])
            out.release()
            print("Saved results to ",vid_name)

    else:
        start = time.time()
        
        for i in range(0,len(test_array)):
            predict_test = np.array(test_array[i]).reshape(1,224,224,3).astype('float32')
            prediction = model.predict(predict_test)
            pred = str(np.where(prediction[0] == np.amax(prediction[0]))[0][0])
            predicted_class = 'C'+str(pred)
            if pred == 0:
                cv2.circle(org_array[i], (10,10), 5, (0,128,0), -1)
                cv2.circle(org_array[i], (10,10), 8, (255,255,255), 1)
                cv2.putText(org_array[i], "Attentive" , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0), 1 )
                # print("Driver Attentive")
            else:
                cv2.circle(org_array[i], (10,10), 5, (0,0,255), -1)
                cv2.circle(org_array[i], (10,10), 8, (255,255,255), 1)
                cv2.putText(org_array[i], "Inattentive - " + str(tags[predicted_class]) , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1 )
                # print("Driver Inattentive -> ",tags[predicted_class])
            prediction_array.append(org_array[i])
        end = time.time()
        fps = len(test_array)/(end - start)
        print("Frames per Second: ",fps)
        if len(test_array) == 1:
            img_name = 'outputs/' + model_name + '_Predicted_image.jpg'
            cv2.imwrite(img_name,prediction_array[0])
            print("Saved results to ",img_name)
        else:
            vid_name = 'outputs/' + model_name + '_Predicted_video.mp4'
            out = cv2.VideoWriter(vid_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, (h,w))
            for i in range(len(prediction_array)):
                out.write(prediction_array[i])
            out.release()
            print("Saved results to ",vid_name)

else:
    print("Error!! {} file doesn't exists. Please provide a valid path".format(path))


    
    
