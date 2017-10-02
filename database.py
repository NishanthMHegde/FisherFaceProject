import numpy as np
import cv2
import os
sample=0
path="user10"
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
for image in imagePaths:
    img = cv2.imread(image)
    cv2.imwrite("User10data/user.10." + str(sample) + ".jpg",img)
    sample = sample +1
