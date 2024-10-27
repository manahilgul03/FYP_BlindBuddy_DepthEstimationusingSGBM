import sys
import cv2
import numpy as np
import mediapipe as mp

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Modified camera setup
cap_right = cv2.VideoCapture(1)
cap_left = cv2.VideoCapture(0)

# Set camera resolution explicitly
WIDTH = 640
HEIGHT = 480
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Adjusted stereo vision parameters
frame_rate = 30  # Reduced frame rate for stability
B = 9  # Distance between cameras [cm]
f = 240  # Adjusted focal length [mm]
alpha = 40  # Camera FOV [degrees]

# Camera calibration matrices (calibrate to get these values)
def get_calibration_matrices():
    camera_matrix_left = np.array([[f * WIDTH, 0, WIDTH / 2],
                                   [0, f * WIDTH, HEIGHT / 2],
                                   [0, 0, 1]], dtype=np.float32)
    camera_matrix_right = np.array([[f * WIDTH, 0, WIDTH / 2],
                                    [0, f * WIDTH, HEIGHT / 2],
                                    [0, 0, 1]], dtype=np.float32)

    dist_coeffs_left = np.zeros((5, 1))
    dist_coeffs_right = np.zeros((5, 1))

    return camera_matrix_left, camera_matrix_right, dist_coeffs_left, dist_coeffs_right

# Function for calculating depth
def calculate_depth(xl, xr, B, f, alpha):
    f_pixel = (WIDTH * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    disparity = xl - xr
    if disparity == 0:
        return None
    depth = (B * f_pixel) / disparity
    if depth < 0 or depth > 300:  # Apply realistic depth constraints
        return None
    return depth

with mp_facedetector.FaceDetection(min_detection_confidence=0.5) as face_detection:
    cam_matrix_left, cam_matrix_right, dist_left, dist_right = get_calibration_matrices()

    while (cap_right.isOpened() and cap_left.isOpened()):
        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()

        if not succes_right or not succes_left:
            break

        # Undistort frames using calibration matrices
        frame_right = cv2.undistort(frame_right, cam_matrix_right, dist_right)
        frame_left = cv2.undistort(frame_left, cam_matrix_left, dist_left)

        # Convert to RGB for face detection
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Detect faces
        results_right = face_detection.process(frame_right_rgb)
        results_left = face_detection.process(frame_left_rgb)

        # Initialize centers and depth as None
        center_left = None
        center_right = None
        depth = None

        if results_right.detections and results_left.detections:
            # Detect the first face in both frames (assuming there's only one)
            detection_right = results_right.detections[0]
            detection_left = results_left.detections[0]

            # Get bounding boxes
            bbox_right = detection_right.location_data.relative_bounding_box
            bbox_left = detection_left.location_data.relative_bounding_box

            # Calculate centers
            center_right = (int((bbox_right.xmin + bbox_right.width / 2) * WIDTH),
                            int((bbox_right.ymin + bbox_right.height / 2) * HEIGHT))
            center_left = (int((bbox_left.xmin + bbox_left.width / 2) * WIDTH),
                           int((bbox_left.ymin + bbox_left.height / 2) * HEIGHT))

            # Draw detections
            mp_draw.draw_detection(frame_right, detection_right)
            mp_draw.draw_detection(frame_left, detection_left)

            # Calculate depth using matched centers
            depth = calculate_depth(center_left[0], center_right[0], B, f, alpha)

        # Display results
        if depth is not None:
            depth_text = f"Distance: {round(depth, 1)} cm"
            cv2.putText(frame_right, depth_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame_left, depth_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame_right, "TRACKING LOST", (75, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_left, "TRACKING LOST", (75, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frames
        cv2.imshow("Right Camera", frame_right)
        cv2.imshow("Left Camera", frame_left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
