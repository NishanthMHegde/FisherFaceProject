#this code is used to fetch each image from each of the different Humans and save only the region of interest(face) in dataset folder
#for training purpose.These images will be trained by the trainer.

import numpy as np
import cv2
import os
path = "User10data"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #use the haar cascade classifiers to detect face
sample=0   #this sample specifies the index of the specific person
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
id =raw_input("Enter ID: ") #used to enter the ID of the person
for image in imagePaths:
    color =cv2.imread(image)


    #convert image to grayscale for detection
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5) #detect faces within grayscale image
    for(x,y,w,h) in faces:
        image = img[y:y+h,x:x+w]
        resized = cv2.resize(image, (640,480)) #resize the images so that both training and test data are of same size
        cv2.imwrite("dataset/User." + str(id) + "." + str(sample) + ".jpg",resized) #write image to the folder in the format of User.id.sample
        cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),3) #draw a rectangle on the colored portion of the image where the face is detected
        cv2.waitKey(100)
    sample = sample +1  #increment the sample
    cv2.imshow('img',color) #display detected portion of the image
    cv2.waitKey(100)


cv2.waitKey(0)
cv2.destroyAllWindows()
