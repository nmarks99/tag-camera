# Tag Camera
AprilTag detection and pose estimation with USB cameras.

**Adapted from:** [Byeongdulee/UR_12idb](https://github.com/Byeongdulee/UR_12idb/tree/main)


## TCP and Camera Coordinate Offsets

All values are relative to the UR tool flange origin. Format: `[x, y, z, rx, ry, rz]` (meters, radians).

### Gripper TCP

```
[0, 0, 0.15, 0, 0, 0]
```

The gripper fingertip center is 150mm along the flange Z axis.

### Camera TCP

```
[0, 0.0433, 0.015, -pi/6, 0, 0]
```

- X: 0mm (no lateral offset)
- Y: +43.3mm from flange center
- Z: +15mm from flange
- RX: -30 degrees (camera is tilted 30 degrees from the tool Z axis toward Y)

### Camera Optical Axis in Tool Coordinates

The camera looks along `[0, sin(30), cos(30)]` = `[0, 0.5, 0.866]` in the tool frame. When the tool Z points straight down, the camera looks 30 degrees off vertical.

### Offset Between Gripper TCP and Camera

- dX: 0mm
- dY: +43.3mm
- dZ: -135mm (camera is 135mm closer to the flange than the gripper tip)
- Straight-line distance: ~120mm (approximate)
