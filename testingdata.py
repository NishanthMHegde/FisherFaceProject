import numpy as np
import cv2
import os
sample=0
path="user15train"
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
for image in imagePaths:
    img = cv2.imread(image)
    cv2.imwrite("user15tdata/user.15." + str(sample) + ".jpg",img)
    sample = sample +1
