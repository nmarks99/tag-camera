from tag_camera import TagCamera
import sys
import time

cam = TagCamera(device_index=1)

if len(sys.argv) != 2:
    print("Please provide ID to locate")
    exit()

id = 0
try:
    id = int(sys.argv[1])
except:
    print(f"Invalid ID {sys.argv[1]}")
    exit()

try:
    while True:
        tag = cam.locate_tag(id)
        if (res):
            x,y = cam.pixel_to_meters(res)
            print(f"Tag {id}:")
            print(f"Center: ({x:+8.2f},{y:+8.2f}) pixels")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass


