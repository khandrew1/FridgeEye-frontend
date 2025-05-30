#!/usr/bin/env python3
import sys
import argparse
import os
import cv2
import base64
import numpy as np

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv

load_dotenv()

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

cred = credentials.Certificate('./serviceAccountKey.json')
fb_db_path = 'https://fridge-eye-ff051-default-rtdb.firebaseio.com/'
firebase_admin.initialize_app(cred, {
    'databaseURL': fb_db_path
})

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
	
# note: to hard-code the paths to load a model, the following API can be used:
net = detectNet(model="/jetson-inference/python/training/detection/ssd/models/fridge-pt2/ssd-mobilenet.onnx", labels="/jetson-inference/python/training/detection/ssd/models/fridge-pt2/labels.txt", 
                input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                threshold=args.threshold)

db_ref = db.reference('/foodItems')

# capture the next image
img = input.Capture()
    
# detect objects in the image (with overlay)
detections = net.Detect(img, overlay=args.overlay)

# print the detections
print("detected {:d} objects in image".format(len(detections)))

for detection in detections:
    # Get the bounding box
    left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

    # Convert jetson_utils image to numpy array
    img_np = np.array(img)

    # Crop the detection from the image
    cropped = img_np[top:bottom, left:right]

    # Encode the cropped image as base64
    _, buffer = cv2.imencode('.jpg', cropped)
    encoded_string = base64.b64encode(buffer).decode('utf-8')

    # Prepare data for Firebase
    data = {
        'itemName': net.GetClassDesc(detection.ClassID),
        'quantity': 1,
        'imageBase64': encoded_string
    }

    db_ref.push(data)
    print(detection)

# for detection in detections:
#     data = {
#         'itemName': net.GetClassDesc(detection.ClassID),
#         'quantity': 1
#     }
#     db_ref.push(data)
#     print(detection)

# render the image
output.Render(img)

# print out performance info
net.PrintProfilerTimes()
