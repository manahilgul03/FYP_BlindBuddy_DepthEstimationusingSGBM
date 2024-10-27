import sys
import cv2
import numpy as np
import time
def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    # Convert focal length f from [mm] to [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print('Left and right camera frames do not have the same pixel width')
        return None

    x_right = right_point[0]
    x_left = left_point[0]

    # Calculate the disparity:
    disparity = x_left - x_right  # Displacement between left and right frames [pixels]

    # Check for zero disparity to avoid division by zero:
    if disparity == 0:
        print('Disparity is zero, depth cannot be calculated')
        return None

    # Calculate depth z:
    zDepth = (baseline * f_pixel) / disparity  # Depth in [cm]

    return zDepth
