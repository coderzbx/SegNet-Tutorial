import cv2
import numpy as np

file_input = '/data/deeplearning/images/zed_pic/0615.jpg'
origin_frame = cv2.imread(filename=file_input)

width = origin_frame.shape[1]
height = origin_frame.shape[0]

img_array = cv2.imencode('.jpg', origin_frame)
img_data = img_array[1]
img_str = img_data.tostring()
with open('./cv2binary.jpg', 'wb') as f:
    f.write(img_str)

image = np.asarray(bytearray(img_str), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
cv2.imwrite('./after.jpg', image)
