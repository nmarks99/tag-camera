from robotiq_tools import WristCamera

cam = WristCamera(device_index=1)
cam.load_calibration()

id = 22
res = cam.locate_tag(id)
if (res):
    print(f"Tag {id} found at distance = {res["distance"]}")
else:
    print("Tag not found")

