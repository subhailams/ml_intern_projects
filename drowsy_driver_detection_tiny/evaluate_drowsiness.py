from utils import *


parser = argparse.ArgumentParser(description='Script to run inference and evaluate driver drowsiness classification. Example: python evaluate_drowsiness.py -m resnet_lstm_large -p inputs/sample_drowsy.mp4')

parser.add_argument('-d','--detect',default = 'mtcnn', help='Model for Face Detection. Select from list ["mtcnn","ultralight"]')
parser.add_argument('-e','--extact',default = 'resnet_lstm_large',help='Model for Feature Extraction. Select from list ["resnet_lstm_large","resnet_lstm_small","mobilenet_lstm"]')
parser.add_argument('-p','--path',default = 'inputs/sample_drowsy.mp4', help='Location of Input video')



class participent:

    def __init__(self, location='None', seq_length=50,face_detect='mtcnn',feature_extract='resnet_lstm_large'):

        self.location = location
        self.det = face_detect
        self.ext = feature_extract
        self.sequence = []
        self.seq_length = seq_length
        self.model = ""
        
        self.ort_session = 0
        self.input_name = ""
        self.extractor = VGGExtractor('resnet50')
        self.detector = ""

        self.prediction = 'Waiting for 50 frames'
        self.alert_score = 0
        self.drowsy_score = 0
        self.det_time = 0
        self.feature_time = 0
        self.prediction_time = 0  
        self.tot_time = 0
        self.current_frame_no = 0

        
    def load_detection_model(self):

        # 1. Ultralight Model Face Detection

        if self.det == 'ultralight':
            onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            onnx.helper.printable_graph(onnx_model.graph)
            predictor = prepare(onnx_model)
            self.ort_session = ort.InferenceSession(onnx_path)
            self.input_name = self.ort_session.get_inputs()[0].name

        # 2. Mtcnn Face Detector 

        if self.det == 'mtcnn':
            self.detector = MTCNN()



    def load_extraction_model(self):
        ## Step 2 Feature Extraction for each image in sequence
        if self.ext == 'resnet_lstm_large' or self.ext == 'resnet_lstm_small':
            # 1. VggFace  Resnet50 Feature Extractor
            self.extractor = VGGExtractor('resnet50')

            # Loading LSTM 2048 model
            if self.ext == 'resnet_lstm_large':
                with open('models/rm.model_vgg_class2.json','r') as f:
                    self.model = model_from_json(f.read())

                self.model.load_weights('models/rm.model_vgg_class2.h5')

            # Loading LSTM 512  model
            if self.ext == 'resnet_lstm_small':
                
                with open('models/model_vggface_resnet50_512_3.json','r') as f:
                    self.model = model_from_json(f.read())

                self.model.load_weights('models/model_vggface_resnet50_512_3.h5')

        if self.ext == 'mobilenet_lstm':
            # 2. MobileFaceNet Feature Extractor
            self.extractor = MobFaceExtractor()

            # Loading model
            with open('models/model_mobface_512_new.json','r') as f:
                self.model = model_from_json(f.read())

            self.model.load_weights('models/model_mobface_512_new.h5')

                
        # Models parameters
        metrics = ['accuracy']
        optimizer = Adam(lr=0.00005)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        
    def get_video_metadata(self, video_path):
        
        c = VideoFileClip(video_path)
        rotation = c.rotation
        fps = c.fps
        c.close()
        
        return rotation,fps

    def ul_preprocess(self,raw_img):
        h, w, _ = raw_img.shape
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        return img

    def predict_sequence(self, sequence):
        lstm_time = time.time()
        predictions_test = self.model.predict(sequence)
        print("LSTM Prediction time:",round(time.time() - lstm_time, 2))
  

        for j in predictions_test:
            print('Alert: ', j[0])
            self.alert_score = round(j[0]*100,2)
            print('Drowsy: ', j[1])
            self.drowsy_score = round(j[1]*100,2)

            if(self.alert_score > self.drowsy_score):
                self.prediction = 'Alert'
                print("Driver is Alert")
                print()

            else:
                self.prediction = 'Drowsy'
                print("Driver is Drowsy")
                print()
    

    def process_frame(self, frame, num):

        return_list = []
        img = frame
        
        if face_detect == 'ultralight':

            # 3. Ultra Light Face Detection
            start_ul = time.time()
            h, w, _ = frame.shape
            img_ul = self.ul_preprocess(frame)
            confidences, boxes = self.ort_session.run(None, {self.input_name: img_ul})
            result, labels, probs = predict(w, h, confidences, boxes, 0.7)
            end_ul = time.time()
            self.det_time = round(end_ul - start_ul, 2)
            
            
            if result != []:
                person = result[0]
                x1, y1, x2, y2 = person[0:]

                return_list += [x1, y1, x2, y2]

                cv2.rectangle(img,(x1,y1),(x2,y2),(80,18,236),2)


                roi_color = img[y1:y2, x1:x2]
                if roi_color.shape[0]<=0 or roi_color.shape[1]<=0:
                    return ['-1' for x in range(4)],img

        if face_detect == 'mtcnn':
    
            # 4. MTCNN Face Detection
            start_mtcnn = time.time()
            result = self.detector.detect_faces(frame)
            end_mtcnn = time.time()
            self.det_time = round(end_mtcnn - start_mtcnn, 2)
            
            
            if result != []:
                person = result[0]
                bounding_box = person['box']
                keypoints = person['keypoints']
                # print(bounding_box)
                x1, y1, width, height = person['box']
                x2, y2 = x1 + width, y1 + height
                return_list += [x1, y1, width, height]

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

                roi_color = img[y1:y2, x1:x2]
                if roi_color.shape[0]<=0 or roi_color.shape[1]<=0:
                    return ['-1' for x in range(4)],img
            
        # Visualising Prediction

        cv2.putText(img, str(num), (x2+10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if(self.prediction == 'Alert'):
            cv2.putText(img, "Alert Score: " +  str(self.alert_score), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,128,0), 2)
            cv2.putText(img, "Drowsy Score: " +  str(self.drowsy_score), (100,140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,128,0), 2)
        elif(self.prediction == 'Drowsy'):
            cv2.putText(img, "Alert Score: " +  str(self.alert_score), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            cv2.putText(img, "Drowsy Score: " +  str(self.drowsy_score), (100,140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)    
        else:
            cv2.putText(img, self.prediction, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
                
                
        # Step 2 Feature Extraction
        if feature_extract == 'resnet_lstm_large' or feature_extract == 'resnet_lstm_small':
            roi_color = cv2.resize(roi_color, (224,224), interpolation = cv2.INTER_AREA)
    

        start_feature = time.time()
        features =  self.extractor.extract(roi_color)
        end_feature = time.time()
        self.feature_time = round(end_feature - start_feature, 2)
        
        self.sequence.append(features)

        return return_list,img

    def process_video(self,image_shape):

        video_path = self.location
        frame_array = []
        # print(video_path)
        # Reporting results
        dropped_frames = []
        img_array = []
        
        
        print("[INFO] starting video file thread...")

        
        if video_path != 'None':
            cap = cv2.VideoCapture(video_path)
            src = video_path
            # Metadata
            video_rotation, video_fps = self.get_video_metadata(video_path)
        else:
            # cap = cv2.VideoCapture(0)
            # cap = cv2.VideoCapture("rtsp://admin:AMmcwCrest@192.168.13.163/")

            video_fps = 30
            video_rotation = 0

        assert(video_fps != 0)
        frame_len = 1/video_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total_frames)

        frame_no = 0
        seq_cnt = 0

        total_start = time.time()

        while(frame_no < total_frames):
            
            ret, frame = cap.read()

          
            frame = imutils.rotate_bound(frame, video_rotation)
            # frame = imutils.resize(frame,width=600)

            frame_row = []
            frame_row.append(video_fps)
            frame_row.append(frame_no)

            result, result_img = self.process_frame(frame,frame_no)       
            frame_row += result
            
            height, width, layers = result_img.shape
            size = (width,height)
        


            if '-1' in frame_row: 
                dropped_frames.append(frame_no)
                print("Face Not Detected\r")
                seq_cnt = -1
                self.sequence = []

            frame_no += 1
            seq_cnt += 1

            if(seq_cnt == self.seq_length):
                print("Predictions: " + str(frame_no))
                seq = []
                seq.append(self.sequence)
                print(np.array(seq).shape)
                self.current_frame_no = frame_no
                self.predict_sequence(np.array(seq))
                self.sequence = []
                seq_cnt = 0
            
            
            frame_array.append(result_img)

            if frame_no == 200:
                break
       
        total_end = time.time()
        print("Face Detection time:",self.det_time)
        print("Feature Extraction time: ",self.feature_time)
        fps = frame_no/(total_end - total_start)
    
        # pathOut = 'outputs/' + feature_extract + '_predicted.avi'
        
        # out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 16, size)
        # for i in range(len(frame_array)):
        #     # writing to a image array
        #     out.write(frame_array[i])
        # out.release()
        # print("Output saved to ->",pathOut)

        cap.release()




args = vars(parser.parse_args())
face_detect = args['detect']
feature_extract = args['extact']
path = args['path']

image_shape = (320, 180)
seq_length = 50
location = path

if os.path.exists(path):

    par = participent(location, seq_length,face_detect, feature_extract)

    print("Loading {} face detection model".format(face_detect));print();

    par.load_detection_model()

    print("Loading {} feature extraction model".format(feature_extract));print();

    par.load_extraction_model()

    print("Predicting... {}".format(path));print()

    par.process_video(image_shape)

else:
    print("Invalid Path!! {} doesnt exists".format(path))
