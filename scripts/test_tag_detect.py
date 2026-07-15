import os
import cv2
import numpy as np
from pupil_apriltags import Detector
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CALIBRATION_FILE = DATA_DIR / "camera_calibration.npz"

# Configuration
device_index = 1
tag_family = "tag36h11"
tag_size_meters = 0.019  # Set to the physical width of your tag in meters (e.g., 50mm = 0.05m)

# 1. Load calibration parameters
if not CALIBRATION_FILE.exists():
    raise FileNotFoundError(f"Missing calibration file: {CALIBRATION_FILE}. Run calibration first.")

with np.load(CALIBRATION_FILE) as data:
    mtx = data['mtx']
    dist = data['dist']

# Format intrinsic parameters for the detector
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
camera_params = [fx, fy, cx, cy]

# 2. Initialize Detector
detector = Detector(
    families=tag_family,
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# 3. Initialize Video Capture
cap = cv2.VideoCapture(device_index)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video device /dev/video{device_index}")

print(f"Starting 3D Pose Estimation. Tag size: {tag_size_meters * 1000:.1f}mm. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags and estimate poses
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size_meters
    )

    for detection in detections:
        tag_id = detection.tag_id
        tvec = detection.pose_t  # 3x1 translation vector
        rvec = detection.pose_R  # 3x3 rotation matrix

        # Extract individual spatial coordinates
        x_m = tvec[0][0]
        y_m = tvec[1][0]
        z_m = tvec[2][0]

        # Calculate absolute Euclidean distance to the tag
        distance = np.linalg.norm(tvec)

        # Print 3D pose metric data to terminal
        print(f"ID: {tag_id:02d} | Distance: {distance:.3f}m | X: {x_m:+.3f}m | Y: {y_m:+.3f}m | Z: {z_m:+.3f}m")

        # Visual feedback: Draw 2D boundaries
        corners = detection.corners.astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw ID text on image
        center = detection.center.astype(int)
        cv2.putText(
            frame, 
            f"ID: {tag_id} ({distance:.2f}m)", 
            (center[0] - 40, center[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            2
        )

    # Render frame
    cv2.imshow("3D Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
