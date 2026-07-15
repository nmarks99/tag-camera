"""Test script for WristCamera interactive viewer."""

from robotiq_tools import WristCamera

def main():
    with WristCamera(device_index=1) as camera:
        try:
            camera.load_calibration()
        except FileNotFoundError:
            print("Warning: No calibration file found, using default focal length")

        camera.show_interactive(detect_tags=True, tag_size=0.019)

if __name__ == "__main__":
    main()
