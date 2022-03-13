import torch
import numpy as np
from mtcnn import MTCNN
import cv2
from PIL import Image
from gender_model import ResNet9, get_default_device, to_device
import torchvision.transforms as tt

# get present device
device = get_default_device()

# load the models parameters
model_gender=ResNet9(3,2)
model_gender.load_state_dict(torch.load('Models\Saved Model\gender_classification.pth',map_location=torch.device('cpu')))

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
        center_img = np.array(Image.fromarray(center_img_k).resize([224, 224]))
        
        # create predictions
        # cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # pil_im = Image.fromarray(cv2_im)
        sex_preds = predict_gender(Image.fromarray(center_img_k))
        # age_preds = age_model.predict(center_img.reshape(1,224,224,3))[0][0]
        age_preds=100

        # output to the cv2
        return_res.append([top, right, bottom, left, sex_preds, age_preds])
        
    return return_res

# preprocess and preduict image from frame
def predict_gender(img):
    transformations=tt.Compose([tt.Resize((64,64)), tt.RandomHorizontalFlip(), tt.ToTensor(),tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    img=transformations(img)
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model_gender(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    # print(preds)
    return gender[preds[0].item()]



# capturing camera frames using cv2
# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color 
    rgb_frame = frame[:, :, ::-1]
    # print(len(rgb_frame))

    # Find all the faces in the current frame of video
    face_locations = detect_face(rgb_frame)

    # Display the results
    for top, right, bottom, left, sex_preds, age_preds in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        sex_text =":"
        text1="Sex: "+sex_preds;
        cv2.putText(frame, text1, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, 'Age: {:.3f}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()