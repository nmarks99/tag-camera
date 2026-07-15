from robotiq_tools import WristCamera
import sys
import time

cam = WristCamera(device_index=1)

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
        res = cam.locate_tag(id)
        if (res):
            x,y = cam.pixel_to_meters(res)
            x *= 1000.0
            y *= 1000.0
            z = res["distance"]*1000.0
            roll, pitch, yaw = res["euler_angles"]

            print(f"Tag {id}:")
            print(f"({x:+8.2f},{y:+8.2f},{z:+8.2f}) mm")
            print(f"({roll:+8.1f},{pitch:+8.1f},{yaw:+8.1f}) deg\n")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass


