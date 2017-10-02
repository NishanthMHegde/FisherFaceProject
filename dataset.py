import numpy as np
import cv2
import os
path = "User10data"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
sample=0
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
id =raw_input("Enter ID: ")
for image in imagePaths:
    color =cv2.imread(image)



    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5)
    for(x,y,w,h) in faces:
        image = img[y:y+h,x:x+w]
        resized = cv2.resize(image, (640,480))
        cv2.imwrite("dataset/User." + str(id) + "." + str(sample) + ".jpg",resized)
        cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.waitKey(100)
    sample = sample +1
    cv2.imshow('img',color)
    cv2.waitKey(100)


cv2.waitKey(0)
cv2.destroyAllWindows()
