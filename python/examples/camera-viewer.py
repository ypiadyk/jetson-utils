#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.utils
import argparse
import cv2

# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")

width, height = 2592, 1458

opt = parser.parse_args()
opt.height = height
opt.width = width
print(opt)

# create display window
display = jetson.utils.glDisplay()

# create camera device
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)

# open the camera for streaming
camera.Open()

# capture frames until user exits
bgr_img = jetson.utils.cudaAllocMapped(width=width, height=height, format='bgr8')


while display.IsOpen():
	image, width, height = camera.CaptureRGBA(timeout=1000, zeroCopy=True)
	print(image)

	# image2 = jetson.utils.cudaAllocMapped(width=width, height=height, format='rgba32')

	# jetson.utils.cudaConvertColor(image, image2)

	# display.RenderOnce(image, width, height)
	# display.SetTitle("{:s} | {:d}x{:d} | {:.1f} FPS".format("Camera Viewer", width, height, display.GetFPS()))


	jetson.utils.cudaConvertColor(image, bgr_img)

	img = jetson.utils.cudaToNumpy(bgr_img)
	# img = jetson.utils.cudaToNumpy(image)
	# img = image
	# img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
	cv2.imshow('my webcam', img)
	# cv2.imshow('my webcam', img[:, :, [2, 1, 0]]/ 255.)
	if cv2.waitKey(1) == 27:
		break  # esc to quit

cv2.destroyAllWindows()
	
# close the camera
camera.Close()

