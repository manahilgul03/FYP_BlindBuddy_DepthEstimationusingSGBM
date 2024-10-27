import cv2
import numpy as np

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

# Load the stereo maps from the file
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

cv_file.release()


def undistortRectify(frameR, frameL):
    """
    This function undistorts and rectifies images using the provided stereo maps.

    Parameters:
        frameR: Image from the right camera.
        frameL: Image from the left camera.

    Returns:
        Tuple containing undistorted and rectified images (right, left).
    """
    # Undistort and rectify images using the stereo maps
    undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Debugging information
    print("Shape of original right frame:", frameR.shape)
    print("Shape of original left frame:", frameL.shape)
    print("Shape of undistorted right frame:", undistortedR.shape)
    print("Shape of undistorted left frame:", undistortedL.shape)

    return undistortedR, undistortedL
