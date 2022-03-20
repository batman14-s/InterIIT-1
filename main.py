import torch
import numpy as np
from mtcnn import MTCNN
import cv2
from PIL import Image
from gender_model import ResNet9, get_default_device, to_device
import torchvision.transforms as tt
from age_model import Age_Model
import scipy
import os
import time
import gdown
from imageai.Detection import ObjectDetection

# get present device
device = get_default_device()

# load the models parameters
model_gender=ResNet9(3,2)
model_gender.load_state_dict(torch.load('Models\Saved Model\gender_classification.pth',map_location=torch.device('cpu')))
model_Age=Age_Model()

# defining output classes
gender=['Female','Male']

# load face detector
detector = MTCNN()

def detect_face(img):
    mt_res = detector.detect_faces(img)
    return_res = []
    
    for face in mt_res:
        x, y, width, height = face['box']
        center = [x+(width/2), y+(height/2)]
        max_border = max(width, height)
        
        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)
        
        # crop the face
        center_img_k = img[top:top+max_border, 
                           left:left+max_border, :]
        center_img = Image.fromarray(center_img_k)
        
        # create predictions
        # cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # pil_im = Image.fromarray(cv2_im)
        sex_preds = predict_gender(center_img)
        age_preds = model_Age.predict_age(center_img_k)

        # output to the cv2
        return_res.append([top, right, bottom, left, sex_preds, age_preds])
        
    return return_res

# object detection using tensorflow
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()


# preprocess and predict image from frame
def predict_gender(img):
    transformations=tt.Compose([tt.Resize((64,64)), tt.RandomHorizontalFlip(), tt.ToTensor(),tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    img=transformations(img)
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model_gender(xb)
    # print(yb)
    probs = scipy.special.softmax(yb.detach().numpy()[0])
    # print(probs)
    confidence=str(round((max(probs))*100,2))+"%" 
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    # print(preds)
    return str(gender[preds[0].item()]) + " (" +str(confidence)+ ")"

def testVideos(vd):
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    # capturing camera frames using cv2
    # Get a reference to webcam 
    video_capture = cv2.VideoCapture(vd)
    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color 
        rgb_frame = frame[:, :, ::-1]
        # print(len(rgb_frame))

        # object setection
        # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "img2.png"), output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)
        detections = detector.detectObjectsFromImage(input_image=rgb_frame, input_type="array" , minimum_percentage_probability=30)

        for eachObject in detections:
            if(eachObject["name"]=="person"):
                print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )

                print("--------------------------------")
                # Find all the faces in the current frame of video
                            # Find all the faces in the current frame of video
                x, y, width, height = eachObject["box_points"]
                center = [x+(width/2), y+(height/2)]
                max_border = max(width, height)
                
                # center alignment
                l = max(int(center[0]-(max_border/2)), 0)
                r = max(int(center[0]+(max_border/2)), 0)
                t = max(int(center[1]-(max_border/2)), 0)
                b = max(int(center[1]+(max_border/2)), 0)

                cv2.rectangle(frame, (l, t), (r, b), (255, 255, 255), 2)
                
                # crop the face
                center_img_k = rgb_frame[t:t+max_border, 
                                l:l+max_border, :]
                face_locations = detect_face(center_img_k)
                # Display the results
                for top, right, bottom, left, sex_preds, age_preds in face_locations:
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                    text1="Gender: "+sex_preds
                    cv2.putText(frame, text1, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                    cv2.putText(frame, 'Age: {:.3f}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

        # time when we finish processing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
    
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)
    
        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
    
        # putting the FPS count on the frame
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "Press 'n' to skip to next video!", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
        # Display the resulting image
        cv2.imshow('Video', frame)
        # print(fps, ",",new_frame_time)
        # Hit 'q' on the keyboard to quit!
        k = cv2.waitKey(1)
        # print("Here!")
        # print(k)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if k == 110:
            print( "Going to next video..........")
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
i=0
for vds in os.listdir("Test Videos/"):
    print("Test Videos/"+vds)
    testVideos("Test Videos/"+vds)
    print("Video ",i," done!")
    i+=1
# testVideos("Test Videos\Cctv2.mp4")