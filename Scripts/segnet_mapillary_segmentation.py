import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time

sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/opt/caffe'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# GPU_ID = 1  # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
# caffe.set_device(GPU_ID)

input_shape = net.blobs['data'].data.shape
# output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

# cv2.namedWindow("Input")
# cv2.namedWindow("SegNet")

start = time.time()

file_input = args.file
origin_frame = cv2.imread(filename=file_input)

width = origin_frame.shape[1]
height = origin_frame.shape[0]

frame = cv2.resize(origin_frame, (input_shape[3], input_shape[2]))
input_image = frame.transpose((2, 0, 1))
# input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
input_image = np.asarray([input_image])
out = net.forward_all(data=input_image)

segmentation_ind = np.squeeze(net.blobs['prob'].data)
segmentation_ind_3ch = np.resize(segmentation_ind, (3, input_shape[2], input_shape[3]))
# segmentation_ind_3ch = segmentation_ind.transpose(1, 2, 0).astype(np.uint8)
segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
# segmentation_rgb = segmentation_rgb.astype(float) / 255

# cv2.imshow("Input", frame)
# cv2.imshow("SegNet", segmentation_rgb)

rgb_frame = cv2.resize(segmentation_rgb, (width, height))

cv2.imwrite('input.jpg', origin_frame)
cv2.imwrite('segnet.png', rgb_frame)

end = time.time()
print('%30s' % 'Processed results in ', str((end - start) * 1000), 'ms\n')

# cv2.destroyAllWindows()

