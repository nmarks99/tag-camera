import cv2
import os

# Configuration
device_index = 1  # Corresponds to /dev/video1
output_dir = "calibration_images"
image_prefix = "calib_img"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        img_name = os.path.join(output_dir, f"{image_prefix}_{img_counter:02d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_counter += 1

# Clean up
cap.release()
cv2.destroyAllWindows()
