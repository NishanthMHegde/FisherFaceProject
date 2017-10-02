import cv2

color = cv2.imread('image.jpg')
col = cv2.resize(color, (400,200))
cv2.imwrite('image2.jpg',col)
