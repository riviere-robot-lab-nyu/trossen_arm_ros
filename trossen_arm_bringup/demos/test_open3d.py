import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        depth_data = np.asanyarray(depth_frame.get_data())
        color_data = np.asanyarray(color_frame.get_data())
        
        # Colorize depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_data, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('Color', color_data)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()