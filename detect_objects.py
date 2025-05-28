from gradio_client import Client, handle_file
import csv
from io import StringIO
import os
from datetime import datetime

class YOLODetectionProcessor:
    def __init__(self):
        self.client = Client("iashin/YOLOv3")
    
    def detect_objects(self, image_path):
        """
        Run object detection on an image using YOLOv3
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Processed detection results
        """
        try:
            print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
            
            # Make API call
            result = self.client.predict(
                source_img=handle_file(image_path),
                api_name="/predict"
            )
            
            print("✅ Detection completed!")
            return self.parse_results(result, image_path)
            
        except Exception as e:
            print(f"❌ Error during detection: {str(e)}")
            return None
    
    def parse_results(self, api_response, original_image_path):
        """Parse the API response into structured data"""
        if not api_response or len(api_response) < 2:
            print("⚠️ No detection results received")
            return None
            
        result_image_path, csv_data = api_response
        
        # Clean CSV data
        csv_clean = csv_data.strip().replace('```', '').strip()
        
        if not csv_clean or csv_clean == 'class,confidence,bx,by,bw,bh':
            print("ℹ️ No objects detected in the image")
            return {
                'original_image': original_image_path,
                'result_image': result_image_path,
                'detected_items': [],
                'total_items': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Parse CSV
        csv_reader = csv.DictReader(StringIO(csv_clean))
        items = []
        
        for row in csv_reader:
            try:
                item = {
                    'class': row['class'],
                    'confidence': float(row['confidence']),
                    'bounding_box': {
                        'center_x': float(row['bx']),
                        'center_y': float(row['by']),
                        'width': float(row['bw']),
                        'height': float(row['bh'])
                    }
                }
                items.append(item)
            except (ValueError, KeyError) as e:
                print(f"⚠️ Skipping malformed detection entry: {e}")
                continue
        
        return {
            'original_image': original_image_path,
            'result_image': result_image_path,
            'detected_items': items,
            'total_items': len(items),
            'timestamp': datetime.now().isoformat()
        }
    
    def print_summary(self, results):
        """Print a formatted summary of detection results"""
        if not results:
            return
            
        print("\n" + "="*60)
        print("🎯 OBJECT DETECTION RESULTS")
        print("="*60)
        print(f"📷 Image: {os.path.basename(results['original_image'])}")
        print(f"📊 Total objects detected: {results['total_items']}")
        print(f"🕒 Processed at: {results['timestamp']}")
        
        if results['total_items'] == 0:
            print("\n🔍 No objects were detected in this image.")
            return
        
        # Group by object class
        object_counts = {}
        for item in results['detected_items']:
            obj_class = item['class']
            if obj_class not in object_counts:
                object_counts[obj_class] = []
            object_counts[obj_class].append(item)
        
        print(f"\n📋 Object Summary:")
        for obj_class, items in object_counts.items():
            avg_confidence = sum(item['confidence'] for item in items) / len(items)
            print(f"   • {obj_class.title()}: {len(items)} detected (avg confidence: {avg_confidence:.1%})")
        
        print(f"\n📍 Detailed Detections:")
        for i, item in enumerate(results['detected_items'], 1):
            bbox = item['bounding_box']
            print(f"   {i}. {item['class'].title()}")
            print(f"      Confidence: {item['confidence']:.1%}")
            print(f"      Position: Center({bbox['center_x']:.3f}, {bbox['center_y']:.3f})")
            print(f"      Size: {bbox['width']:.3f} × {bbox['height']:.3f}")
            print()
        
        print(f"🖼️ Annotated result saved to: {results['result_image']}")
        print("="*60)
    
    def get_objects_by_class(self, results, object_class):
        """Get all detections of a specific object class"""
        if not results or not results['detected_items']:
            return []
        return [item for item in results['detected_items'] if item['class'].lower() == object_class.lower()]
    
    def get_high_confidence_objects(self, results, min_confidence=0.8):
        """Get objects above a confidence threshold"""
        if not results or not results['detected_items']:
            return []
        return [item for item in results['detected_items'] if item['confidence'] >= min_confidence]
    
    def export_to_json(self, results, output_file=None):
        """Export results to JSON file"""
        if not results:
            return None
            
        import json
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"detection_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results exported to: {output_file}")
        return output_file

# Example usage
def main():
    # Initialize detector
    detector = YOLODetectionProcessor()
    
    # Run detection on your image
    image_path = 'InsideFridge6.jpg'
    
    print("🚀 Starting YOLOv3 Object Detection...")
    results = detector.detect_objects(image_path)
    
    if results:
        # Print comprehensive summary
        detector.print_summary(results)
        
        # Example queries
        print("\n🔍 CUSTOM QUERIES:")
        
        # Find all bottles
        bottles = detector.get_objects_by_class(results, 'bottle')
        if bottles:
            print(f"🍼 Found {len(bottles)} bottles")
        
        # Find high-confidence detections
        high_conf = detector.get_high_confidence_objects(results, min_confidence=0.95)
        if high_conf:
            print(f"⭐ {len(high_conf)} high-confidence detections (>95%)")
        
        # Export results
        detector.export_to_json(results)
        
        # Return results for further processing
        return results
    
    return None

if __name__ == "__main__":
    # Run the detection
    detection_results = main()
    
    # You can continue processing the results here
    if detection_results:
        print(f"\n✨ Detection complete! Found {detection_results['total_items']} objects.")
        
        # Example: Access specific data
        for item in detection_results['detected_items']:
            print(f"- {item['class']}: {item['confidence']:.1%}")