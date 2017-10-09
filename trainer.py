import os
import numpy as np
import cv2
from PIL import Image

recognizer = cv2.createFisherFaceRecognizer() #used to invoke the FisherFaceRecognizer
path = "dataSet"

def getImagesWithPath(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces =[]
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')  #convert the image obtained to grayscale if not already in grayscale
        faceNp = np.array(faceImg,'uint8') #Convert the face image into a numpy array of datatyp unsigned int
        ID = int(os.path.split(imagePath)[-1].split('.')[1])  #split the image name to obtain user ID
        faces.append(faceNp)
        IDs.append(ID)
        print ID
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return faces,IDs
faces,IDs = getImagesWithPath(path)  #retrieve the faces and IDS from imagepath
recognizer.train(faces,np.array(IDs))  #train each image with the corresponding ID
recognizer.save('recognizer/trainingData.yml') #save the trained result into the recognizer folder
cv2.destroyAllWindows()
