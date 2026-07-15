import cv2
import numpy as np

# Load the calibration data
with np.load('camera_calibration.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Load a calibration image (preferably one with noticeable lens curve at the edges)
img = cv2.imread('calibration_images/calib_img_05.jpg')
h, w = img.shape[:2]

# Refine the camera matrix based on free scaling parameter (alpha = 0 preserves all pixels)
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# Crop the image to remove black edges if desired (using the ROI)
x, y, w_roi, h_roi = roi
cropped_img = undistorted_img[y:y+h_roi, x:x+w_roi]

# Display comparison
cv2.imshow('Original Image (Distorted)', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
