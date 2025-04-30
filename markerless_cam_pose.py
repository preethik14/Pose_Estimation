import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

model = YOLO('/home/rbccps/synthetic_data/runs/pose/train6/weights/best.pt') 

camera_matrix = np.array([[429.9493431,    0.,         657.10347183],
 [  0.,         427.32633934, 413.00928701],
 [  0.,           0.,           1.        ]])

dist_coeffs = np.array([[ 0.07899657, -0.07846187, -0.02638088,  0.03126975,  0.01879872]])

keypoints_3d = np.array([[2.741665, 3.733775, 5.745445], 
[2.741663, 3.733785, 5.745437], 
[3.471888, 1.516255, 6.049605], 
[4.259058, -0.2029828, 6.926162], 
[3.763209, -0.002970581, 5.652419], 
[3.640406, -0.2890424, 5.062484], 
[3.206568, 1.035543, 4.887618], 
[-6.825047, -14.7194, 8.878982], 
[4.89535, -1.148479, 7.997787], 
[5.522304, -3.825602, 7.626967], 
[-0.0735219, 1.99391, 9.664993], 
[-0.7999998, 0.2919891, 6.252188], 
[8.588893, -4.099284, 2.494826], 
[8.178126, -4.099284, 3.798842], 
[-1.603574, -4.099284, 0.8678982], 
[-4.903562, -4.099285, 4.250894], 
[-38.22319, -1.954104, 5.653496], 
[-38.22319, -1.954104, 4.247582], 
[-2.003329, -2.061641, 4.539421], 
[-2.003329, -1.873348, 6.066184], 
[-0.8017714, -0.3009222, 6.216832], 
[-2.003329, -1.21307, 4.097618], 
[-38.22319, -2.578677, 4.996911], 
[-5.133297, -4.099285, 2.297769], 
[7.78628, -4.099285, 4.939589], 
[2.39828, 5.25881, 5.996537], 
[3.534632, 4.706014, 8.838669], 
[3.771632, 4.617208, 9.45314], 
[-3.349279, -14.7194, 5.222944], 
[-5.104284, -14.7194, 7.029432]])

def rotation_vector_to_euler(rvec):
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Extract Euler angles (RPY) from rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6  # Check if the rotation is singular

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0  # Yaw is not well-defined in singular cases

    # Convert from radians to degrees
    roll, pitch, yaw = np.degrees([roll, pitch, yaw])

    return roll, pitch, yaw

def estimate_pose(keypoints_2d, keypoints_3d):
    # Convert keypoints to numpy arrays
    keypoints_2d = np.array(keypoints_2d)
    keypoints_3d = np.array(keypoints_3d)/1000
    # Perform PnP
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs)
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return translation_vector, rotation_vector
def camera_callback(frame):
    """Process frame, detect keypoints, and estimate pose."""
    try:
        # Run inference
        results = model(frame, verbose=False)[0]  # Get first result object
        if results.keypoints is None:
            print("No keypoints detected.")
            return

        keypoints = results.keypoints.xy.cpu().numpy()  # Extract (x, y) coordinates
        for kp in keypoints:
            x, y = int(kp[0][0]), int(kp[0][1])
            # print("x:", x)
            # print("y:", y)
            # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Estimate pose
        translation, rotation = estimate_pose(keypoints, keypoints_3d)
       
        x, y, z = translation.flatten()
        print("xyz", x,y,z)
        roll, pitch, yaw = rotation_vector_to_euler(rotation)
        print("rpy", roll, pitch, yaw)

        # Display on frame
        cv2.putText(frame, f"X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if translation is not None and rotation is not None:
            print(f"Translation: {translation.flatten()}, Rotation: \n{rotation}")

        # Display results
        cv2.imshow("KeyPoints & Pose", frame)
        cv2.waitKey(10)

    except Exception as e:
        print(f"Error in processing: {e}")
    
video_path = "/home/rbccps/synthetic_data/cam2_diadem.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Perform inference
    camera_callback(frame)