import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CALIBRATION_IMAGES_DIR = DATA_DIR / "calibration_images"

# Configuration
device_index = 1  # Corresponds to /dev/video1
output_dir = CALIBRATION_IMAGES_DIR
image_prefix = "calib_img"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(device_index)

if not cap.isOpened():
    print(f"Error: Could not open video device at /dev/video{device_index}")
    exit()

print("Instructions:")
print("  Press 's' to save a calibration frame.")
print("  Press 'q' to quit.")

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the live feed
    cv2.imshow("Calibration Capture", frame)

    # Wait for keypress (1 ms delay)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Exiting capture script.")
        break
    elif key == ord('s'):
        # Generate file path
        img_name = output_dir / f"{image_prefix}_{img_counter:02d}.jpg"
        cv2.imwrite(str(img_name), frame)
        print(f"Saved: {img_name}")
        img_counter += 1

# Clean up
cap.release()
cv2.destroyAllWindows()
