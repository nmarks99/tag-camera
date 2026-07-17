"""Configuration and path management for tag_camera."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_calibration_file() -> Path:
    """Get the default calibration file path."""
    return get_data_dir() / "camera_calibration.npz"


def get_calibration_images_dir() -> Path:
    """Get the calibration images directory."""
    return get_data_dir() / "calibration_images"
