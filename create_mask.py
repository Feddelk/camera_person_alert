import cv2
import numpy as np
import json
from pathlib import Path

# Try to load config.py
try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

class MaskCreator:
    def __init__(self):
        self.points = []
        self.drawing = False
        self.current_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback for mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point added: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points.pop()
                print(f"Last point removed. Remaining points: {len(self.points)}")
    
    def create_mask(self, rtsp_url="0"):
        """Create detection zone mask for camera"""
        print("\n" + "="*60)
        print("🎨 DETECTION ZONE CREATOR")
        print("="*60)
        print("\nInstructions:")
        print("  • LEFT CLICK: Add point")
        print("  • RIGHT CLICK: Remove last point")
        print("  • Key 'S': Save mask")
        print("  • Key 'C': Clear all points")
        print("  • Key 'Q': Exit without saving")
        print("="*60 + "\n")
        
        # Open camera
        if rtsp_url == "0" or rtsp_url == "":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return None
        
        print("✓ Camera connected\n")
        
        # Get a frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            cap.release()
            return None
        
        self.current_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Create window
        cv2.namedWindow('Create Detection Zone')
        cv2.setMouseCallback('Create Detection Zone', self.mouse_callback)
        
        while True:
            # Update frame from camera
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame.copy()
            
            display = self.current_frame.copy()
            
            # Draw points
            for i, point in enumerate(self.points):
                cv2.circle(display, point, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw lines between points
            if len(self.points) > 1:
                for i in range(len(self.points)):
                    cv2.line(display, self.points[i], 
                            self.points[(i+1) % len(self.points)], 
                            (0, 255, 255), 2)
            
            # Draw area if enough points
            if len(self.points) >= 3:
                # Create temporary mask
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(self.points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
                # Semi-transparent overlay
                overlay = display.copy()
                overlay[mask > 0] = [0, 255, 0]
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            
            # On-screen instructions
            cv2.putText(display, f"Points: {len(self.points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "S=Save | C=Clear | Q=Exit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Create Detection Zone', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if len(self.points) >= 3:
                    # Save mask
                    mask_data = {
                        'points': self.points,
                        'width': w,
                        'height': h,
                        'rtsp_url': rtsp_url
                    }
                    
                    mask_file = Path("detection_mask.json")
                    with open(mask_file, 'w') as f:
                        json.dump(mask_data, f, indent=2)
                    
                    print(f"\n✓ Mask saved to: {mask_file}")
                    print(f"✓ Points defined: {len(self.points)}")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return self.points
                else:
                    print("⚠️ You need at least 3 points to create a zone")
            
            elif key == ord('c') or key == ord('C'):
                self.points = []
                print("🗑️ Points cleared")
            
            elif key == ord('q') or key == ord('Q'):
                print("❌ Cancelled by user")
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        return self.points

def load_mask():
    """Load saved mask"""
    mask_file = Path("detection_mask.json")
    if mask_file.exists():
        with open(mask_file, 'r') as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    import sys
    
    # Try to use RTSP URL from config.py
    if HAS_CONFIG and hasattr(config, 'RTSP_URL'):
        rtsp_url = config.RTSP_URL
        print(f"📹 Using RTSP URL from config.py: {rtsp_url}")
    else:
        rtsp_url = input("Enter RTSP URL (or press Enter for webcam): ").strip()
        if not rtsp_url:
            rtsp_url = "0"
    
    creator = MaskCreator()
    points = creator.create_mask(rtsp_url)
    
    if points:
        print("\n✅ Mask created successfully!")
        print("Now start the application normally with: python app.py")
    else:
        print("\n❌ No mask was created")
