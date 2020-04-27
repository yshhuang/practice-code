import cv2
import numpy as np

# Read images : obj image will be cloned into im
obj = cv2.imread("/Volumes/develop/tool/flag.jpg")
im = cv2.imread("/Volumes/develop/tool/hhh.jpg")
H, W = obj.shape[:2]
# obj = cv2.resize(obj, (W, H - 150), cv2.INTER_CUBIC)
# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the obj in the im
width, height, channels = im.shape

center = (int(height / 2), int(width / 2))

# Seamlessly clone obj into im and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
cv2.imwrite("/Volumes/develop/tool/normal_clone.jpg", normal_clone)
cv2.imwrite("/Volumes/develop/tool/mixed_clone.jpg", mixed_clone)
