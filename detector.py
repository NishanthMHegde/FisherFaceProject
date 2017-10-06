import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.createFisherFaceRecognizer()
recognizer.load('recognizer/trainingData.yml')
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)




while True:
    color = cv2.imread('user10tdata/user.10.2.jpg')


    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),3)

        p_image = img[y:y+h,x:x+w]
        img = cv2.resize(p_image,(640,480))
        gaus = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,0)
        cv2.imshow('gaus',gaus)
        id,conf = recognizer.predict(gaus)

        cv2.cv.PutText(cv2.cv.fromarray(color),str(id),(x,y+h),font,255)






    cv2.imshow('img',color)
    cv2.waitKey(1)
    k =cv2.waitKey(30) &0xff
    if k==27:
        break




cv2.destroyAllWindows()
