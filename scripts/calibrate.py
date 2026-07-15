import numpy as np
import cv2
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CALIBRATION_IMAGES_DIR = DATA_DIR / "calibration_images"
CALIBRATION_OUTPUT = DATA_DIR / "camera_calibration.npz"

# Define checkerboard dimensions (number of internal corners: columns-1, rows-1)
# Example: An 8x6 square board has 7x5 internal corners.
CHECKERBOARD = (16, 22)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,4,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Path to the directory containing calibration images
images = glob.glob(str(CALIBRATION_IMAGES_DIR / "*.jpg"))

if not images:
    raise FileNotFoundError("No images found in the specified directory.")

gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

if len(objpoints) > 0:
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # Save camera matrix and distortion coefficients to .npz file
    np.savez(CALIBRATION_OUTPUT, mtx=mtx, dist=dist)
    print(f"Calibration successful. Data saved to '{CALIBRATION_OUTPUT}'.")
else:
    print("Calibration failed. No checkerboard patterns detected.")

# Calculate overall RMS and list of per-view errors
total_error = 0
per_view_errors = []

for i in range(len(objpoints)):
    # Project 3D points back to the image plane
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Ensure both arrays are 2D arrays of float32 with shape (N, 2)
    pts1 = imgpoints[i].astype(np.float32).reshape(-1, 2)
    pts2 = imgpoints2.astype(np.float32).reshape(-1, 2)

    # Calculate absolute difference (L2 norm)
    error = cv2.norm(pts1, pts2, cv2.NORM_L2) / len(pts2)
    per_view_errors.append(error)
    total_error += error ** 2

rms_error = np.sqrt(total_error / len(objpoints))

# Extract intrinsic values
fx, fy = mtx[0, 0], mtx[1, 1]
cx, cy = mtx[0, 2], mtx[1, 2]
aspect_ratio = fx / fy

# Expected image center
ideal_cx = gray_shape[0] / 2
ideal_cy = gray_shape[1] / 2
cx_offset = abs(cx - ideal_cx)
cy_offset = abs(cy - ideal_cy)
print("=" * 50)
print("           CAMERA CALIBRATION METRICS           ")
print("=" * 50)
print(f"Overall RMS Reprojection Error : {rms_error:.4f} pixels")
print(f"Number of calibration images used: {len(objpoints)}")
print("-" * 50)
print(f"Focal Length (fx, fy)          : ({fx:.2f}, {fy:.2f})")
print(f"Pixel Aspect Ratio             : {aspect_ratio:.4f} (Ideal: ~1.00)")
print("-" * 50)
print(f"Optical Center (cx, cy)        : ({cx:.2f}, {cy:.2f})")
print(f"Expected Center                : ({ideal_cx:.1f}, {ideal_cy:.1f})")
print(f"Center Offset (X, Y)           : ({cx_offset:.2f} px, {cy_offset:.2f} px)")
print("-" * 50)
print("Distortion Coefficients (k1, k2, p1, p2, k3):")
print(f"  Radial (k1, k2, k3)          : {dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][4]:.5f}")
print(f"  Tangential (p1, p2)          : {dist[0][2]:.5f}, {dist[0][3]:.5f}")
print("-" * 50)
print("Per-Image Reprojection Errors:")
for idx, err in enumerate(per_view_errors):
    status = "OK" if err < 1.0 else "BAD (Consider removing)"
    print(f"  Image {idx:02d}: {err:.4f} pixels -- {status}")
print("=" * 50)
