## **InterIIT Bosch's Age And Gender Detection**

![alt text](https://www.cio.com/wp-content/uploads/2021/12/intro_ts_ai_ml_by-monsitj-getty-images_2400x1600-100853894-orig.jpg?quality=50&strip=all)

### PROBLEM STATEMENT

### DESCRIPTION

The scenes obtained from a surveillance video are usually with low resolution. Most of the scenes captured by a static camera are with minimal change of background. Objects in outdoor surveillance are often detected in far-fields. Most existing digital video surveillance systems rely on human observers for detecting specific activities in a real-time video scene. However, there are limitations in the human capability to monitor simultaneous events in surveillance displays. Hence, human motion analysis in automated video surveillance has become one of the most active and attractive research topics in the area of computer vision and pattern recognition.

### PAIN POINT

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: How do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution.

### PROBLEM STATEMENT

Build a solution to estimate the gender and age of people from a surveillance video feed (like mall, retail store, hospital etc.). Consider low resolution cameras as well as cameras put at a height for surveillance. Hint: Develop a pipeline to apply super resolution algorithms for people detected and then apply gender/age estimation algorithms. Problem approach:
1. Object Detection: imageai for object detection, SRGAN algorithms for image enhancement, faceai for face detection.
2. Classification and estimation: Deep learning and neural networks

Domains: Super Resolution GAN, DL, Image Processing

### PIPELINE
![pipeline_bosch jpg](https://user-images.githubusercontent.com/101988266/159222526-3b8a8576-b21f-4e64-aae9-cc74defee417.jpg)

As can be seen from the pipeline, 
1. The first step we do is extract video frames from the video and pass them through the Imageai library of python which does object detection.
2. From all the objects detected by applying imageai, we extract the body of all individuals which are categorized as ‘Person’ category.
3. After extraction of images of individuals we apply sr-gan model on them for body feature extraction to increase the accuracy of our prediction. This process results in high resolution images.
4. From the bodies extracted, face detection is done using MTCNN which stands for Multi-task Cascaded Convolutional Networks and is a framework used for face extraction.
5. Age prediction is done using deepface and gives the exact age of the person.
6. Gender prediction is done using a resnet9 model built and trained on commercially available facial image dataset obtained from Kaggle (link to dataset)
7. After parsing through the whole video a csv is generated as output according to the required submission format.

### OUR APPROACH
- We use the [imageai package](https://github.com/OlafenwaMoses/ImageAI) for object detection, which returns the various objects found in the frames provided. We take the 'person' objects from the returned objects and use them to determine age and gender. Using RetinaNet, YOLOv3, and TinyYOLOv3 trained on the COCO dataset, ImageAI enables object detection, video detection, video object tracking, and image predictions trainings.

- For increasing the image resolution we are using tensorflows pre-trained model [ESRGAN](https://www.tensorflow.org/hub/tutorials/image_enhancing).
To give a superior flow gradient at the microscopic level, the model uses the Residual-in-Residual block as a basic convolution block instead of a basic residual network or plain convolution trunk. Furthermore, the model lacks a batch normalization layer in the generator to prevent the smoothing out of image artifacts. This allows the ESRGAN to provide images with a better representation of the image artifacts sharp edges.The architecture visualization is as follows:
![esrgan_arch](https://user-images.githubusercontent.com/101988266/159240238-3696e1f2-748e-4bac-84e8-336b57661da2.png)
- We have implemented our own SRGAN model as well. That can also be used in place of ESRGAN model of tensorflow.

- [MTCNN](https://github.com/ipazc/mtcnn) stands for Multi-task Cascaded Convolutional Networks, which was created as a solution for both face identification and face alignment. We are using this for face detection from the individual images detected. The method entails three levels of convolutional networks that can recognise faces and landmarks such as the eyes, nose, and mouth. It uses a shallow CNN in the first step to quickly generate candidate windows. It refines the recommended candidate windows in the second stage using a more complicated CNN. Finally, it employs a third CNN, which is more complicated than the others, to improve the result and output face landmark positions in the third step.

- The dataset we utilised was taken from Kaggle and is commercially available. It is made up of male and female facial photos that have been divided into 116 categories (of age). We created a Resnet model for gender prediction that was trained on this kaggle dataset and gives over 95% accuracy on the test set for gender. The Dataset used for Gender Classification can be found at the following link: [Dataset Gender](https://drive.google.com/drive/folders/1-tkzrNYITjm6MpMaefMmJ7OaBVfdglJD?usp=sharing)

- We have used the [Deepface library](https://github.com/serengil/deepface) for predicting the age, which outputs the exact age. Deepface is a Python framework for facial recognition and attribute analysis. The fundamental components of this library are inspired by Keras and Tensorflow. It's a hybrid face recognition framework that combines state-of-the-art analysis models like VGG-Face into a single package and its facial recognition accuracy reaches upto 97 percent. We have used it for age prediction.
- We could've used Deepface Library for gender prediction as well for higher accuracy.

![Age and gender prediction](https://user-images.githubusercontent.com/101988266/159292929-a2513d72-ea67-4c1c-98db-2a5dc5d7670d.gif)

### Dependencies for ImageAi

1. Python 3.7.5 (>3.6.0 and <3.9.0)
2. Tensorflow 2.4.0
3. OpenCV
4. Keras 2.4.3
 
### Steps to run

1. Firstly downgrade your system python to 3.7.5 version for Imageai to work smoothly

2. To prevent manual labour, run following command in the terminal with required Python version (3.7.5) to install every required dependencies for this project.

```
cd InterIIT
pip install -r requirements.txt
```

3. Download the following files from following links into the same directory.

[Yolo.h5](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5)

Save following videos in a folder in same directory named "Test Videos"

[Test Videos](https://drive.google.com/drive/folders/1WNrJGt2lxAPomkkx363eJ8_hB7Tydaw8?usp=sharing)

4. Then finally run following code and you will be good to go :)

```
python main.py
```
5. Press "n" on keyboard to skip to next test video. You may also put your own test videos in the folder.

Thanks & Regards

Team

### Steps to run

1. Clone this repo locally by using fllowing command in your terminal.

```
git clone git@github.com:yashcode00/InterIIT.git
```
2. Then, install all the dependicies by using the follwing commands on same terminal.

```
cd InterIIT
pip install -r requirements.txt
```

3. Then finally run following code and you will be good to go :)

```
python main.py
```
### How to run / test local videos
1. Just paste your test video in the "Test Videos" folder in the same "InterIIT" Directory.
2. Just follow step 3 above to obtain the csv and a live video demostration.
3. Be patient, it may take long time.

Thanks & Regards

Team
