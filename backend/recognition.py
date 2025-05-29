#!/usr/bin/env python3

import argparse
import datetime
import sys
import os

from jetson_inference import detectNet
from jetson_utils import (videoSource, saveImage, Log,
                          cudaAllocMapped, cudaCrop, cudaDeviceSynchronize)

# parse the command line
parser = argparse.ArgumentParser(description="Capture a single frame, locate objects using a detection DNN, "
                                             "save snapshots of detections and the full frame.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream (e.g., /dev/video0, csi://0)")
# The 'output' argument is no longer used for a display stream, but kept for potential parsing by underlying utilities
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream (unused for display in this script)")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are: 'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--snapshots", type=str, default="images/test/detections_single_shot", help="output directory for detection snapshots and full frame")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in snapshot filenames")
parser.add_argument("--full-frame-filename", type=str, default="full_frame_capture.jpg", help="Filename for the full captured frame. A timestamp will be prepended.")


try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# make sure the snapshots dir exists
os.makedirs(args.snapshots, exist_ok=True)
Log.Info(f"Snapshots will be saved to: {os.path.abspath(args.snapshots)}")

# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# create video sources
# The 'argv' is passed to allow videoSource to parse its own relevant options
input_stream = videoSource(args.input, argv=sys.argv)

# Capture the single image
Log.Info("Capturing a single frame...")
img = input_stream.Capture(timeout=5000) # Add a timeout, e.g., 5000ms

if img is None: # timeout or error
    Log.Error("Failed to capture image from input source.")
    sys.exit(1)

Log.Info(f"Frame captured: {img.width}x{img.height}, format={img.format}")

# Detect objects in the image (with overlay if specified)
# The 'img' object will be modified with overlays if args.overlay is not 'none'
detections = net.Detect(img, overlay=args.overlay)

# Print the detections
Log.Info("Detected {:d} objects in image".format(len(detections)))

# Generate a timestamp for this capture session
capture_timestamp = datetime.datetime.now().strftime(args.timestamp)

# Save snapshots of individual detections
for idx, detection in enumerate(detections):
    Log.Info(f"Detection {idx}: {detection}")
    # Define Region of Interest for cropping
    roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
    
    # Ensure ROI coordinates are valid
    if roi[0] >= roi[2] or roi[1] >= roi[3]:
        Log.Warning(f"Skipping invalid ROI for detection {idx}: {roi}")
        continue

    try:
        # Allocate memory for the snapshot
        snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
        # Crop the detection from the main image
        cudaCrop(img, snapshot, roi)
        cudaDeviceSynchronize() # Ensure cropping is complete
        
        # Define filename for the cropped detection
        snapshot_filename = os.path.join(args.snapshots, f"{capture_timestamp}-detection-{idx:02d}-{net.GetClassDesc(detection.ClassID)}.jpg")
        saveImage(snapshot_filename, snapshot)
        Log.Success(f"Saved detection snapshot: {snapshot_filename}")
        del snapshot # Release memory
    except Exception as e:
        Log.Error(f"Error processing/saving detection {idx}: {e}")


# Save the full frame (which includes overlays if enabled)
full_frame_base_name, full_frame_ext = os.path.splitext(args.full_frame_filename)
if not full_frame_ext: # if no extension, default to .jpg
    full_frame_ext = ".jpg"
full_frame_output_filename = os.path.join(args.snapshots, f"{capture_timestamp}-{full_frame_base_name}{full_frame_ext}")

try:
    saveImage(full_frame_output_filename, img)
    Log.Success(f"Saved full frame: {full_frame_output_filename}")
except Exception as e:
    Log.Error(f"Error saving full frame: {e}")

# Print out performance info
net.PrintProfilerTimes()

# Clean up
del img
del input_stream
del net

Log.Info("Single shot detection and snapshot process complete.")
