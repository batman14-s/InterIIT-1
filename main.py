import torch
import pandas as pd
import numpy as np
from mtcnn import MTCNN
import cv2
from PIL import Image
from gender_model import ResNet9, get_default_device, to_device
import torchvision.transforms as tt
from age_model import Age_Model
from scipy import special
import os
import time
import gdown
from imageai.Detection import ObjectDetection
import matplotlib.pyplot as plt
from image_enhancing import enhance

# get present device
device = get_default_device()

# load the models parameters
model_gender=ResNet9(3,2)
model_gender.load_state_dict(torch.load('Models\Saved Model\gender_classification.pth',map_location=torch.device('cpu')))
model_Age=Age_Model()

enhancer=enhance()

# defining output classes
gender=['Female','Male']

# load face detector
detectorf = MTCNN()

def detect_face(img):
    mt_res = detectorf.detect_faces(img)
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
        center_img_k=enhancer.enhanceit(center_img_k)
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
    probs = special.softmax(yb.detach().numpy()[0])
    # print(probs)
    confidence=str(round((max(probs))*100,2))+"%" 
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    # print(preds)
    return str(gender[preds[0].item()]) + " (" +str(confidence)+ ")"

def testVideos(vd,i):
    dict={"frame":[],"personid":[],"bb_xmin":[],"bb_ymin":[],"bb_height":[],"bb_width":[],"age_min":[],"age_max":[],"age_actual":[],"gender":[]}
    frameno=0
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    # capturing camera frames using cv2
    # Get a reference to webcam 
    video_capture = cv2.VideoCapture(vd)
    while video_capture.isOpened():
        frameno+=1
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # frame=enhancer.enhanceit(frame)
        # Convert the image from BGR color (which OpenCV uses) to RGB color 
        # rgb_frame = frame[:, :, ::-1]
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(type(rgb_frame))
            # print(len(rgb_frame))

            # object setection
            # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "img2.png"), output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)
            detections = detector.detectObjectsFromImage(input_type="array", input_image=rgb_frame , output_image_path=os.path.join(execution_path , "image.jpg")) # For numpy array input type
            person=0
            dy = 25
            for eachObject in detections:
                flag=1
                if(eachObject["name"]=="person"):
                    print("Person detected!!!!!!!!!!!")
                    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
                    print("--------------------------------")
                    # Find all the faces in the current frame of video
                                # Find all the faces in the current frame of video
                    x, y, width, height = eachObject["box_points"]
                    print(x,",",y,",",width,",",height)
                    print(rgb_frame.shape)

                    cv2.rectangle(frame, (x, max(y-dy,0)), (width, height), (0, 0, 255), 2)
                    
                    # croping the detected body out of the frame
                    center_img_k = rgb_frame[max(y-dy,0):height, x:width, :]
                    # plt.imshow(Image.fromarray(center_img_k))
                    # plt.show()
                    # print(center_img_k.shape)
                    # center_img_k=enhancer.enhanceit(center_img_k)
                    # plt.imshow(Image.fromarray(center_img_k))
                    # plt.show()
                    # cv2.imwrite("original.png",(center_img_k))
                    face_locations = detect_face(center_img_k)
                    # Display the results
                    for top, right, bottom, left, sex_preds, age_preds in face_locations:
                        person+=1
                        dict["age_actual"].append(age_preds)
                        dict["gender"].append(sex_preds)
                        dict["frame"].append(frameno)
                        dict["personid"].append(person)
                        dict["bb_xmin"].append(min(x,width))
                        dict["bb_ymin"].append(min(y,height))
                        dict["bb_height"].append(abs(y-height))
                        dict["bb_width"].append(abs(x-width))
                        dict["age_min"].append("Nan")
                        dict["age_max"].append("Nan")
                        print(person)
                        # Draw a box around the face
                        left+=x
                        top+=y
                        right+=x
                        bottom+=y
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                        text1="Gender: "+sex_preds
                        cv2.putText(frame, text1, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                        cv2.putText(frame, 'Age: {:.3f}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        flag+=1

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
            # cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press 'n' to skip to next video!", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                
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
        except:
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    db=pd.DataFrame(dict)
    db.to_csv("Video"+str(i)+".csv")
i=0
for vds in os.listdir("Test Videos/"):
    print("Test Videos/"+vds)
    testVideos("Test Videos/"+vds,i)
    print("Video ",i," done!")
    i+=1
# testVideos("Test Videos/"+"WhatsApp Video 2022-03-21 at 10.57.10 PM.mp4",i)