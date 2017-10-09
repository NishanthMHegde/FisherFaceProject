import numpy as np
import cv2
import os
sample=0
path="user10"  #used to specify the path from where pgm pictures are imported
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
for image in imagePaths: #loop through each imagepath
    img = cv2.imread(image)
    cv2.imwrite("User10data/user.10." + str(sample) + ".jpg",img) #write each image as jpeg
    sample = sample +1 #increment sample num
