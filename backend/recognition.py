#!/usr/bin/env python3
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# (Based on detectnet-snap.py, modified for single-shot capture, Firebase, & .env)
#
# ... (rest of the copyright and permission notice remains the same) ...
#
# This script captures a single frame, detects objects, saves detected objects
# to individual images (optional), saves the full frame with detections (optional),
# and sends detection data including Base64 cropped images to Firebase Realtime DB.
# Firebase credentials can be provided via a .env file or command-line arguments.
#
# Example usage with .env file:
# python3 detectnet-single-shot-firebase-env.py <input_URI>
#
# python3 detectnet-single-shot.py \
#   <input_URI>
#

import argparse
import datetime
import sys
import os

# For loading .env file
from dotenv import load_dotenv

# Jetson Inference and Utils
from jetson_inference import detectNet
from jetson_utils import (videoSource, saveImage, Log,
                          cudaAllocMapped, cudaCrop, cudaDeviceSynchronize,
                          cudaToNumpy)

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# Image processing and Base64 encoding
import base64
import io
from PIL import Image
import numpy as np

# --- Load Environment Variables from .env file ---
load_dotenv()


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Capture a single frame, detect objects, save snapshots, "
                                             "and send data to Firebase Realtime Database. "
                                             "Firebase settings can be sourced from a .env file.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + Log.Usage())

# Standard arguments
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream (e.g., /dev/video0, csi://0)")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream (unused for display in this script)")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

# File snapshot arguments
parser.add_argument("--snapshots", type=str, default="images/test/detections_firebase_env", help="output directory for local image snapshots")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in filenames and data")
parser.add_argument("--full-frame-filename", type=str, default="full_frame_capture.jpg", help="Base filename for the full captured frame saved locally.")
parser.add_argument("--save-local-snapshots", action='store_true', help="Enable saving cropped detection snapshots locally.")
parser.add_argument("--save-local-fullframe", action='store_true', help="Enable saving the full frame locally.")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# --- Determine Firebase Configuration (CLI > .env > Default) ---
# Path to Firebase service account key JSON file
fb_creds_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
# Firebase Realtime Database URL
fb_db_url = os.getenv("FIREBASE_DATABASE_URL")
# Firebase Realtime Database path
fb_db_path = os.getenv("FIREBASE_DB_PATH", "foodItems") # Default to "foodItems" if not in CLI or .env


# --- Directory Setup for Local Snapshots ---
if args.save_local_snapshots or args.save_local_fullframe:
    os.makedirs(args.snapshots, exist_ok=True)
    Log.Info(f"Local snapshots will be saved to: {os.path.abspath(args.snapshots)}")

# --- Firebase Initialization ---
firebase_admin_initialized = False
if fb_creds_path and fb_db_url:
    if not os.path.exists(fb_creds_path):
        Log.Error(f"Firebase credentials file not found at: {fb_creds_path}")
        Log.Error("Firebase integration will be disabled for this run.")
    else:
        try:
            cred = credentials.Certificate(fb_creds_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': fb_db_url
            })
            Log.Info(f"Successfully initialized Firebase Admin SDK with DB URL: {fb_db_url}")
            firebase_admin_initialized = True
        except Exception as e:
            Log.Error(f"Failed to initialize Firebase Admin SDK: {e}")
            Log.Error("Firebase integration will be disabled for this run.")
else:
    Log.Warning("Firebase credentials path or DB URL not provided (checked CLI args and .env). Firebase integration disabled.")


# --- Load Network and Video Source ---
net = detectNet(args.network, sys.argv, args.threshold)
input_stream = videoSource(args.input, argv=sys.argv)

# --- Capture Frame ---
Log.Info("Capturing a single frame...")
img = input_stream.Capture(timeout=10000)

if img is None:
    Log.Error("Failed to capture image from input source.")
    if firebase_admin_initialized:
        firebase_admin.delete_app(firebase_admin.get_app()) # Clean up Firebase app
    sys.exit(1)
Log.Info(f"Frame captured: {img.width}x{img.height}, format={img.format}")

# --- Detect Objects ---
detections = net.Detect(img, overlay=args.overlay)
Log.Info("Detected {:d} objects in image".format(len(detections)))

capture_timestamp_obj = datetime.datetime.now()
capture_timestamp_str = capture_timestamp_obj.strftime(args.timestamp)

# --- Process Detections ---
for idx, detection in enumerate(detections):
    Log.Info(f"Processing Detection {idx}: {detection}")
    
    roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
    if roi[0] >= roi[2] or roi[1] >= roi[3]:
        Log.Warning(f"Skipping invalid ROI for detection {idx}: {roi}")
        continue

    base64_snapshot_str = None
    pil_img_format_ext = 'jpeg' # Default extension
    try:
        snapshot_img = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
        cudaCrop(img, snapshot_img, roi)
        cudaDeviceSynchronize()
        
        numpy_img = cudaToNumpy(snapshot_img)
        if numpy_img.dtype == np.float32 or numpy_img.dtype == np.float16:
            numpy_img = (numpy_img * 255).astype(np.uint8)

        pil_img = Image.fromarray(numpy_img)
        img_byte_buffer = io.BytesIO()
        
        pil_save_format = 'JPEG'
        if pil_img.mode == 'RGBA':
             pil_save_format = 'PNG'
             pil_img_format_ext = 'png'
        elif pil_img.mode == 'L':
             pil_img_format_ext = 'jpeg' # or 'png' if preferred for grayscale

        pil_img.save(img_byte_buffer, format=pil_save_format, quality=85 if pil_save_format=='JPEG' else None)
        img_bytes = img_byte_buffer.getvalue()
        base64_snapshot_str = base64.b64encode(img_bytes).decode('utf-8')
        Log.Info(f"Successfully encoded snapshot for detection {idx} to Base64.")

        if args.save_local_snapshots:
            snapshot_filename = os.path.join(args.snapshots, f"{capture_timestamp_str}-detection-{idx:02d}-{net.GetClassDesc(detection.ClassID)}.{pil_img_format_ext}")
            saveImage(snapshot_filename, snapshot_img)
            Log.Info(f"Saved local detection snapshot: {snapshot_filename}")
        del snapshot_img

    except Exception as e:
        Log.Error(f"Error processing/converting snapshot for detection {idx}: {e}")
        if 'snapshot_img' in locals() and snapshot_img is not None:
             del snapshot_img

    # --- Prepare and Push Data to Firebase ---
    if firebase_admin_initialized and base64_snapshot_str:
        try:
            class_label = net.GetClassDesc(detection.ClassID)
            item_name_fb = f"{class_label} - Instance {detection.Instance} ({(detection.Confidence * 100):.1f}%)"

            firebase_data = {
                'itemName': item_name_fb,
                'quantity': 1,
                'details': {
                    'classLabel': class_label,
                    'instance': detection.Instance,
                    'confidence': round(detection.Confidence, 4),
                    'captureTimestamp': capture_timestamp_obj.isoformat(),
                    'sourceInput': args.input,
                    'networkUsed': args.network,
                    'boundingBox': {
                        'left': detection.Left, 'top': detection.Top,
                        'right': detection.Right, 'bottom': detection.Bottom,
                        'width': detection.Width, 'height': detection.Height,
                        'area': detection.Area,
                        'centerX': detection.Center[0], 'centerY': detection.Center[1]
                    }
                },
                'croppedImageBase64': base64_snapshot_str
            }
            
            db_ref = db.reference(fb_db_path) # Use determined path
            db_ref.push(firebase_data)
            Log.Success(f"Pushed detection '{item_name_fb}' data to Firebase path '{fb_db_path}'.")

        except Exception as e:
            Log.Error(f"Error pushing data to Firebase for detection {idx}: {e}")
    elif not base64_snapshot_str and firebase_admin_initialized:
         Log.Warning(f"Base64 snapshot not available for detection {idx}, skipping Firebase push for this item.")

# --- Save Full Frame Locally (Optional) ---
if args.save_local_fullframe:
    full_frame_base_name, full_frame_ext = os.path.splitext(args.full_frame_filename)
    if not full_frame_ext: full_frame_ext = ".jpg"
    full_frame_output_filename = os.path.join(args.snapshots, f"{capture_timestamp_str}-{full_frame_base_name}{full_frame_ext}")
    try:
        saveImage(full_frame_output_filename, img)
        Log.Success(f"Saved local full frame: {full_frame_output_filename}")
    except Exception as e:
        Log.Error(f"Error saving local full frame: {e}")

# --- Performance Info & Cleanup ---
net.PrintProfilerTimes()
if img: del img
if input_stream: del input_stream
if net: del net

# Clean up Firebase app instance if it was initialized
if firebase_admin_initialized:
    try:
        firebase_admin.delete_app(firebase_admin.get_app())
        Log.Info("Cleaned up Firebase app instance.")
    except Exception as e:
        Log.Error(f"Error cleaning up Firebase app: {e}")

Log.Info("Single shot detection, snapshot, and Firebase push process complete.")
