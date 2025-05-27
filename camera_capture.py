#!/usr/bin/env python3
#!chmod +x camera_capture.py

import cv2
import time
import os
from datetime import datetime
import argparse
import signal
import sys

class CameraCapture:
    def __init__(self, save_directory="/home/jetson/images", width=1920, height=1080):
        self.save_directory = save_directory
        self.width = width
        self.height = height
        self.camera = None
        self.running = True
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\nShutting down gracefully...")
        self.running = False
    
    def setup_camera(self):
        """Initialize the camera with optimal settings for IMX219"""
        # GStreamer pipeline for IMX219 camera
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink"
        )
        
        try:
            self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.camera.isOpened():
                print("Error: Could not open camera with GStreamer pipeline")
                print("Trying fallback method...")
                # Fallback to direct camera access
                self.camera = cv2.VideoCapture(0)
                
            if not self.camera.isOpened():
                raise Exception("Could not initialize camera")
                
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            return False
    
    def capture_image(self):
        """Capture a single image and save it"""
        if not self.camera or not self.camera.isOpened():
            print("Camera not available")
            return False
        
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture image")
            return False
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self.save_directory, filename)
        
        # Save the image
        success = cv2.imwrite(filepath, frame)
        
        if success:
            print(f"Image saved: {filepath}")
            return True
        else:
            print(f"Failed to save image: {filepath}")
            return False
    
    def run_continuous_capture(self, interval_seconds=60):
        """Run continuous image capture at specified intervals"""
        if not self.setup_camera():
            return
        
        print(f"Starting continuous capture every {interval_seconds} seconds")
        print(f"Images will be saved to: {self.save_directory}")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                self.capture_image()
                
                # Wait for the specified interval
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description="Continuous camera capture for Jetson Nano with IMX219")
    parser.add_argument("--directory", "-d", 
                       default="/home/jetson/images",
                       help="Directory to save images (default: /home/jetson/images)")
    parser.add_argument("--interval", "-i", 
                       type=int, default=60,
                       help="Capture interval in seconds (default: 60)")
    parser.add_argument("--width", "-w", 
                       type=int, default=1920,
                       help="Image width (default: 1920)")
    parser.add_argument("--height", "-h", 
                       type=int, default=1080,
                       help="Image height (default: 1080)")
    parser.add_argument("--single", "-s", 
                       action="store_true",
                       help="Capture a single image and exit")
    
    args = parser.parse_args()
    
    # Initialize camera capture
    capture = CameraCapture(
        save_directory=args.directory,
        width=args.width,
        height=args.height
    )
    
    if args.single:
        # Single capture mode
        if capture.setup_camera():
            capture.capture_image()
            capture.cleanup()
    else:
        # Continuous capture mode
        capture.run_continuous_capture(args.interval)

if __name__ == "__main__":
    main()