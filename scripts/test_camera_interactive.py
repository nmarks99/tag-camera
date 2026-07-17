"""Test script for TagCamera interactive viewer."""

from tag_camera import TagCamera

def main():
    with TagCamera(device_index=1) as camera:
        camera.show_interactive(detect_tags=True, tag_size=0.019)

if __name__ == "__main__":
    main()
