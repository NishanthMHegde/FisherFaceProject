import numpy as np
import cv2

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gaus = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,11)

cv2.imshow('gaus',gaus)

cv2.waitKey(0)
cv2.destroyAllWindows()
