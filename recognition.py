#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect.")
parser.add_argument("--topK", type=int, default=5, help="top k")
args = parser.parse_args()


# load an image (into shared CPU/GPU memory)
img = jetson.utils.loadImage(args.filename)

# load the recognition network
net = jetson.inference.imageNet(args.network)

# classify the image
class_idx, confidence = net.Classify(img)

# find the object description
class_desc = net.GetClassDesc(class_idx)

predictions = net.Classify(img, topK=args.topK)

for n, (classID, confidence) in enumerate(predictions):
   classLabel = net.GetClassLabel(classID)
   confidence *= 100.0
   print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

# print out the result
# print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
