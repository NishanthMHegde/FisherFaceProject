import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.createFisherFaceRecognizer()
recognizer.load('recognizer/trainingData.yml') #load the trainingdata into the recognizer
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4) #font to use to display user ID

while True:
    color = cv2.imread('user10tdata/user.10.2.jpg') #read the testing image

    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) #onvert the image to grayscale
    faces = face_cascade.detectMultiScale(img,1.3,5) #detect the face portion within the grayscale

    for(x,y,w,h) in faces: # starting coordinate, width and height of face
        cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),3)

        p_image = img[y:y+h,x:x+w]  #store the region of interest in a temporary variable
        img = cv2.resize(p_image,(640,480)) #resize the image to match that of the training data
        gaus = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,0) #apply Gaussian adaptive threshold to the image

        cv2.imshow('gaus',gaus)
        id,conf = recognizer.predict(gaus) #use the predictor to predict the userID

        cv2.cv.PutText(cv2.cv.fromarray(color),str(id),(x,y+h),font,255) #display the user ID that has been retrieved

    cv2.imshow('img',color)
    cv2.waitKey(1)
    k =cv2.waitKey(30) &0xff
    if k==27:
        break




cv2.destroyAllWindows()
