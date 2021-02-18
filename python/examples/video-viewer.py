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

import jetson.inference
import jetson.utils

import argparse
import sys
import time
import cv2

# parse command line
# parser = argparse.ArgumentParser(description="View various types of video streams", 
#                                  formatter_class=argparse.RawTextHelpFormatter, 
#                                  epilog=jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

# parser.add_argument("--input_URI", type=str, default="csi://0", help="URI of the input stream")
# parser.add_argument("--output_URI", type=str, default="", nargs='?', help="URI of the output stream")
# parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
# parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

# try:
# 	opt = parser.parse_args()
# except:
# 	print("")
# 	parser.print_help()
# 	sys.exit(0)


# (2592, 1458, 30)
# create video sources & outputs
args = ["--input-width=2592", "--input-height=1944", "--input-rate=15", "--num-buffers=8", "--flip-method=rotate-180"]
# args = ["--input-width=2592", "--input-height=1458", "--input-rate=30"]
# args = ["--input-width=1280", "--input-height=720", "--input-rate=30"]

# args.extend(["--input-flip=2"])

input = jetson.utils.videoSource("csi://0", argv=args)
# output = jetson.utils.videoOutput("", argv=[])

# net = jetson.inference.detectNet("ssd-mobilenet-v1", threshold=0.5)
# net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)

# capture frames until user exits
i, t0 = None, 0
old_timestamp = 0
# while output.IsStreaming():
while True:
	# image = input.Capture(format="rgba32f")
	# image = input.Capture(format="rgb8")
	image = input.Capture(format="raw")

	if i is None:
		i = 0
		t0 = time.time()
	else:
		i += 1
		print("\nTotal FPS:", i / (time.time() - t0))

	if i % 10 == 0:
		timestamp = image.timestamp
		print(image, timestamp, (timestamp-old_timestamp) / 1.e+9)
		old_timestamp = timestamp

	# output.Render(image)
	# output.SetStatus("Video Viewer | {:d}x{:d} | {:.1f} FPS".format(image.width, image.height, output.GetFrameRate()))
	
	# t1 = time.time()
	# detections = net.Detect(image, overlay="")
	# print("Inference time:", time.time() - t1)
	# print("Network FPS", net.GetNetworkFPS(), "Detected:", len(detections))
	# net.PrintProfilerTimes()

	t1 = time.time()
	img = jetson.utils.cudaToNumpy(image)
	print(img.shape, img.dtype)
	img = img.ravel()
	n = img.shape[0]
	img = img[:n*2//3].reshape((1944, 2592))
	print("Extraction time:", (time.time() - t1))
	cv2.imshow('my webcam', img)
	# cv2.imshow('my webcam', img[:, :, ::-1])
	# cv2.imshow('my webcam', img[:, :, [2, 1, 0]]/ 255.)
	# cv2.imshow('my webcam', img[:, :, ::-1])
	if cv2.waitKey(1) == 27:
		break  # esc to quit

	time.sleep(0.001)

cv2.destroyAllWindows()


