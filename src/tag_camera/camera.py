import numpy as np
import cv2
import math
from pathlib import Path
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

DEFAULT_RESOLUTION = (640, 480)
DEFAULT_FOCAL_LENGTH = 586
DEFAULT_FOCUS = 475
FOCUS_RANGE = (350, 600)
FOCUS_STEP = 5
BLUR_THRESHOLD = 510
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE = 0.019

FOCUS_PRESETS = {
    0: 400,
    1: 450,
    2: 475,
    3: 500,
    4: 525,
    5: 550,
    6: 600,
    7: 650,
}


def variance_of_laplacian(image: np.ndarray) -> float:
    """Compute Laplacian variance for focus quality assessment."""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry(image: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """
    Check if image is blurry using Laplacian variance on central region.

    Only checks a 128×72 central region for speed.
    """
    size = image.shape
    ysize = 72
    xsize = 128
    y0 = int(size[0] / 2 - ysize)
    y1 = int(size[0] / 2 + ysize - 1)
    x0 = int(size[1] / 2 - xsize)
    x1 = int(size[1] / 2 + xsize - 1)

    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm < threshold


class TagCamera:
    """
    Interface for Robotiq Wrist Camera.

    Provides camera control, calibration loading, focus management,
    and interactive viewing with AprilTag detection.
    """

    def __init__(
        self,
        device_index: int = 1,
        resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        tag_family: str = DEFAULT_TAG_FAMILY,
        tag_size: float = DEFAULT_TAG_SIZE
    ):
        """
        Initialize Wrist Camera.

        Args:
            device_index: Video device index (default: 1 for /dev/video1)
            resolution: Camera resolution as (width, height)
            tag_family: AprilTag family for detection (default: "tag36h11")
            tag_size: Physical tag size in meters (default: 0.019)
        """
        self.device_index = device_index
        self.resolution = resolution
        self.tag_family = tag_family
        self.tag_size = tag_size
        self.vidcap: Optional[cv2.VideoCapture] = None
        self._detector = None

        self.camera_f = DEFAULT_FOCAL_LENGTH
        self.intrinsic_mtx: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        self._last_frame: Optional[np.ndarray] = None

        self.open()
        try:
            self.load_calibration()
        except FileNotFoundError:
            pass
        self.disable_autofocus()
        self.set_focus(DEFAULT_FOCUS)

    def open(self):
        """Open the camera device and configure settings."""
        if self.vidcap is not None and self.vidcap.isOpened():
            return

        self.vidcap = cv2.VideoCapture(self.device_index)
        if not self.vidcap.isOpened():
            raise RuntimeError(f"Cannot open camera at /dev/video{self.device_index}")

        self.vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        for _ in range(5):
            self.vidcap.read()

        actual_width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale_factor = actual_width / DEFAULT_RESOLUTION[0]
        self.camera_f = DEFAULT_FOCAL_LENGTH * scale_factor

    def close(self):
        """Release the camera device."""
        if self.vidcap is not None:
            self.vidcap.release()
            self.vidcap = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        cv2.destroyAllWindows()

    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self.vidcap is not None and self.vidcap.isOpened()

    @property
    def calibration_loaded(self) -> bool:
        """Check if calibration data is loaded."""
        return self.intrinsic_mtx is not None

    def _get_detector(self):
        """Get or create the shared AprilTag detector."""
        if self._detector is None:
            from pupil_apriltags import Detector
            self._detector = Detector(
                families=self.tag_family,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
        return self._detector

    def _get_camera_params(self) -> list:
        """Get [fx, fy, cx, cy] for pose estimation."""
        if self.calibration_loaded:
            fx = self.intrinsic_mtx[0, 0]
            fy = self.intrinsic_mtx[1, 1]
            cx = self.intrinsic_mtx[0, 2]
            cy = self.intrinsic_mtx[1, 2]
        else:
            w, h = self.resolution
            fx = fy = self.camera_f
            cx = w / 2.0
            cy = h / 2.0
        return [fx, fy, cx, cy]

    def capture(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.

        Returns:
            Tuple of (success, frame)
        """
        if not self.is_open:
            return False, None

        ret, frame = self.vidcap.read()
        if ret:
            self._last_frame = frame
        return ret, frame

    def set_focus(self, value: int):
        """
        Set manual focus value.

        Args:
            value: Focus value (typically 350-600 for Robotiq camera)
        """
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        self.vidcap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.vidcap.set(cv2.CAP_PROP_FOCUS, value)

    def enable_autofocus(self):
        """Enable camera autofocus."""
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        self.vidcap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def disable_autofocus(self):
        """Disable camera autofocus (manual mode)."""
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        self.vidcap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def get_focus(self) -> int:
        """Get current focus value."""
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        return int(self.vidcap.get(cv2.CAP_PROP_FOCUS))

    def scan_focus(self) -> int:
        """
        Scan through focus range to find sharpest focus.

        Returns:
            Best focus value found
        """
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        print("Scanning focus range...")
        best_focus = DEFAULT_FOCUS

        for focus_val in range(FOCUS_RANGE[0], FOCUS_RANGE[1] + 1, FOCUS_STEP):
            self.set_focus(focus_val)
            cv2.waitKey(100)

            ret, img = self.capture()
            if ret and not is_blurry(img):
                best_focus = focus_val
                print(f"Found good focus at {best_focus}")
                break

        self.set_focus(best_focus)
        return best_focus

    def load_calibration(self, path: Optional[Path] = None):
        """
        Load camera calibration from .npz file.

        Args:
            path: Path to calibration file. If None, uses default location.
        """
        if path is None:
            from tag_camera.config import get_calibration_file
            path = get_calibration_file()

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        data = np.load(path)
        self.intrinsic_mtx = data['mtx']
        self.dist_coeffs = data['dist']

        fx = self.intrinsic_mtx[0, 0]
        fy = self.intrinsic_mtx[1, 1]
        self.camera_f = (fx + fy) / 2

        #  print(f"Calibration loaded from {path}")
        #  print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}, avg={self.camera_f:.2f}")

    def estimate_tag_distance(self, corners: np.ndarray, tag_size: float) -> float:
        """
        Estimate distance to AprilTag using pixel-based method.

        Does not require calibration to be loaded.

        Args:
            corners: 4×2 array of tag corner coordinates
            tag_size: Physical tag size in meters

        Returns:
            Estimated distance in meters
        """
        edge_lengths = []
        for k in range(4):
            ind1 = k % 4
            ind2 = (k + 1) % 4
            x0, y0 = corners[ind1]
            x1, y1 = corners[ind2]
            d = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            edge_lengths.append(d)

        mean_edge_pixels = np.mean(edge_lengths)
        distance = tag_size / mean_edge_pixels * self.camera_f
        return distance

    def pixel_to_meters(
        self,
        data: list | dict,
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to real-world offset from optical center.

        If calibration is loaded, undistorts the pixel coordinates first
        and uses the calibrated intrinsics. Otherwise falls back to the
        approximate camera_f and image center.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            distance: Distance to the object in meters (depth along optical axis)

        Returns:
            Tuple of (x_meters, y_meters) offset from optical center
        """

        pixel_x = 0.0
        pixel_y = 0.0
        distance = 0.0
        if (isinstance(data, list)):
            pixel_x = data[0]
            pixel_y = data[1]
            distance = data[2]
        elif (isinstance(data, dict)):
            pixel_x = data["center"][0]
            pixel_y = data["center"][1]
            distance = data["distance"]
        else:
            raise RuntimeError("Unsupported pixel data type provided")


        if self.calibration_loaded:
            pts = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
            undistorted = cv2.undistortPoints(pts, self.intrinsic_mtx, self.dist_coeffs, P=self.intrinsic_mtx)
            ux, uy = undistorted[0][0]

            fx = self.intrinsic_mtx[0, 0]
            fy = self.intrinsic_mtx[1, 1]
            cx = self.intrinsic_mtx[0, 2]
            cy = self.intrinsic_mtx[1, 2]

            x_m = (ux - cx) / fx * distance
            y_m = (uy - cy) / fy * distance
        else:
            ret, frame = self.capture()
            if ret:
                h, w = frame.shape[:2]
            else:
                w, h = self.resolution
            cx = w / 2.0
            cy = h / 2.0

            x_m = (pixel_x - cx) / self.camera_f * distance
            y_m = (pixel_y - cy) / self.camera_f * distance

        return x_m, y_m

    def locate_tag(
        self,
        tag_id: int,
        tag_size: Optional[float] = None
    ) -> Optional[dict]:
        """
        Locate a specific AprilTag by ID.

        Captures a fresh frame, runs detection with pose estimation,
        and returns info for the tag matching the given ID.

        Args:
            tag_id: The AprilTag ID to search for
            tag_size: Physical tag size in meters. If None, uses self.tag_size.

        Returns:
            Dict with keys:
                "center": pixel coordinates (ndarray)
                "distance": distance in meters (float)
                "corners": 4x2 pixel coordinates (ndarray)
                "tf": 4x4 transformation matrix from Camera origin to AprilTag (ndarray)
            Returns None if tag not found.
        """
        if not self.is_open:
            return None

        if tag_size is None:
            tag_size = self.tag_size

        ret, frame = self.capture()
        if not ret:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = self._get_detector()
        camera_params = self._get_camera_params()
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size
        )

        for detection in detections:
            if detection.tag_id == tag_id:

                # Need to flip about the Z axis
                R_flip = np.diag([-1, -1, 1])
                R_corrected = R_flip @ detection.pose_R
                t_corrected = R_flip @ detection.pose_t

                tf = np.block([
                    [R_corrected, t_corrected.reshape(3, 1)],
                    [np.array([[0, 0, 0, 1]])]
                ])

                return {
                    "center": detection.center,
                    "distance": self.estimate_tag_distance(detection.corners, tag_size),
                    "corners": detection.corners,
                    "tf" : tf,
                }

        return None

    def show_interactive(
        self,
        detect_tags: bool = True,
        tag_size: Optional[float] = None
    ):
        """
        Display interactive camera viewer with AprilTag detection.

        Keyboard controls:
            0-7: Set preset focus values
            a: Enable autofocus
            x: Disable autofocus (manual mode)
            s: Scan for best focus
            p: Print tag info to terminal
            h: Show help
            q or ESC: Quit

        Args:
            detect_tags: Enable AprilTag detection and overlay
            tag_size: Physical tag size in meters. If None, uses self.tag_size.
        """
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        if tag_size is None:
            tag_size = self.tag_size

        detector = None
        if detect_tags:
            try:
                detector = self._get_detector()
            except ImportError:
                print("Warning: pupil_apriltags not available, tag detection disabled")
                detect_tags = False

        first_frame = True
        current_detections = []

        print("Interactive camera viewer started")
        print("Press 'h' for help, 'q' or ESC to quit")

        while True:
            ret, frame = self.capture()
            if not ret:
                print("Failed to capture frame")
                break

            if first_frame:
                h, w = frame.shape[:2]
                print(f"Camera frame size: [{w}, {h}]")
                first_frame = False

            current_detections = []
            if detect_tags and detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                camera_params = self._get_camera_params()
                detections = detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=tag_size
                )

                for detection in detections:
                    corners = detection.corners.astype(int)
                    for j in range(4):
                        cv2.line(frame, tuple(corners[j]), tuple(corners[(j + 1) % 4]), (0, 255, 0), 2)

                    tag_pos = detection.center
                    tag_dist = self.estimate_tag_distance(detection.corners, tag_size)
                    r = Rotation.from_matrix(detection.pose_R)
                    euler = r.as_euler('xyz', degrees=True)
                    current_detections.append((detection, tag_pos, tag_dist, euler))

            cv2.imshow('Robotiq Wrist Camera', frame)

            key = cv2.waitKey(20) & 0xFF

            if key in [ord(str(i)) for i in range(8)]:
                preset_num = int(chr(key))
                focus_val = FOCUS_PRESETS[preset_num]
                self.set_focus(focus_val)
                print(f"Focus set to {focus_val}")

            elif key == ord('a'):
                self.enable_autofocus()
                print("Autofocus enabled")

            elif key == ord('x'):
                self.disable_autofocus()
                print("Autofocus disabled (manual mode)")

            elif key == ord('s'):
                best_focus = self.scan_focus()
                print(f"Focus scan complete: {best_focus}")

            elif key == ord('p'):
                if len(current_detections) > 0:
                    sorted_dets = sorted(current_detections, key=lambda d: d[0].tag_id)
                    print("=" * 70)
                    print(f"Detected {len(sorted_dets)} tag(s)")
                    print("-" * 70)
                    for detection, tag_pos, tag_dist, euler in sorted_dets:
                        roll, pitch, yaw = euler
                        print(
                            f"  ID:{detection.tag_id:3d}  "
                            f"Dist:{tag_dist*1000:6.1f}mm  "
                            f"Roll:{roll:+7.1f}  "
                            f"Pitch:{pitch:+7.1f}  "
                            f"Yaw:{yaw:+7.1f}"
                        )
                    print("=" * 70)
                else:
                    print("No tags detected")

            elif key == ord('h'):
                print("\n" + "=" * 50)
                print("HELP - Keyboard Controls")
                print("=" * 50)
                print("Focus controls:")
                print("  0-7       : Set preset focus values (400-650)")
                print("  a         : Enable autofocus")
                print("  x         : Disable autofocus (manual mode)")
                print("  s         : Scan for best focus")
                print("\nTag detection:")
                print("  p         : Print tag info to terminal")
                print("\nGeneral:")
                print("  h         : Show this help")
                print("  q or ESC  : Quit")
                print("=" * 50 + "\n")

            elif key == ord('q') or key == 27:
                print("Exiting interactive viewer")
                break

        cv2.destroyAllWindows()
