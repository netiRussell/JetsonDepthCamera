import depthai as dai
import numpy as np

# Set the threshold distance in meters
d = 2.0  # You can adjust this value as needed

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutDepth.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputDepth(True)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    # Get calibration data
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(
        dai.CameraBoardSocket.LEFT, 640, 400
    )
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    # Output queue to receive depth frames
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    # Convert threshold distance to millimeters
    d_mm = d * 1000  # Convert meters to millimeters

    while True:
        inDepth = qDepth.get()  # Blocking call

        depthFrame = inDepth.getFrame()  # Depth frame as numpy array

        # Create a mask where depth is less than d_mm
        mask = depthFrame < d_mm

        # Get indices of points within the threshold distance
        indices = np.argwhere(mask)

        # Get depth values of those points
        Z = depthFrame[mask].astype(np.float32)  # Depth values in mm

        # Get pixel coordinates
        u = indices[:, 1].astype(np.float32)  # x-coordinates (width)
        v = indices[:, 0].astype(np.float32)  # y-coordinates (height)

        # Compute real-world coordinates using the pinhole camera model
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Convert from millimeters to meters
        X /= 1000.0
        Y /= 1000.0
        Z /= 1000.0

        # Combine X, Y, Z into coordinates
        coordinates = np.column_stack((X, Y, Z))

        # Output the number of points detected
        print(f"Number of points within {d} meters: {coordinates.shape[0]}")

        # Add any additional processing or visualization here

        # Break condition (optional)
        # if some_condition:
        #     break
