#------------camera facing machines----------------


import cv2
import numpy as np
import csv
from ultralytics import YOLO
from collections import deque

# Load YOLO model
model = YOLO('/media/rbccps/Kingston/synthetic_data_testing/synthetic_unity.pt') 

# Camera parameters
# camera_matrix = np.array([[662.19945478, 0., 627.14119636],
#                           [0., 672.85037723, 425.88803109],
#                           [0., 0., 1.]])

camera_matrix = np.array([[1.49056927e+03, 0.00000000e+00, 9.50965040e+02],
 [0.00000000e+00, 1.47928863e+03, 6.07568546e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

# dist_coeffs = np.array([[0.76900098], [-2.16149455], [-0.06720713], [0.02200486], [1.79411061]])
dist_coeffs = np.array([[-0.4459706,   0.2564508,  -0.00615072,  0.00251301, -0.1173754]], dtype=np.float32) 

# # Homography matrix
# H = np.array([[ 0.08780717, -0.02541965, -0.77610436],
#  [-0.00177898, -0.03833974, -0.04485501],
#  [ 0.00712767,  0.11447755, 1.        ]])

H = np.array([[ 8.56270560e-02, -3.31266448e-02, -7.85436653e-01],
 [-8.78688147e-04, -4.32422686e-02, -1.21589944e-01],
 [ 1.46985379e-02,  1.24201567e-01,  1.00000000e+00]], dtype=np.float32)


# 3D keypoints
keypoints_3d = np.array([[0.4051, -0.0776, 0.358], [0.398, 0.0499, 0.345], [0.512, 0.214, 0.265], [0.508, -0.18, 0.265],[0.361, -0.013, 0.367],
                         [0.3853, 0.121, 0.3441], [0.386, -0.15, 0.345], [0.3355, -0.1374, 0.3872], [0.3407, 0.1228, 0.3867],
                         [-0.5291, -0.203, 0.2645], [-0.525, 0.197, 0.271], [-0.3996, 0.123, 0.3398], [-0.394, -0.1464, 0.3425],
                         [-0.241, -0.337, 0.3878], [-0.13, -0.337, 0.387], [-0.0104, -0.336, 0.3858], [0.14, -0.329, 0.389], 
                         [0.244, -0.331, 0.386], [-0.216, 0.321, 0.391], [-0.103, 0.324, 0.388], [0.0163, 0.318, 0.3852], [0.129, 0.3141, 0.3796],
                         [0.244, 0.3196, 0.3835], [0.0005, 0.2003, 0.9691], [-0.013, -0.247, 0.615], [0, -0.02, 0.612], [-0.011, -0.251, 0.832],
                         [-0.021, -0.0201, 0.8254], [-0.009, 0.223, 0.829], [0, 0.223, 0.615]])  # Keep your keypoints_3d array as is

# keypoints_3d = np.array([
#     [-0.52675329,  1.44664772, -3.95430743],
#     [ 1.15107476,  0.19984933, -4.11893241],
#     [-3.15467571, -1.91868933, -2.48073073],
#     [ 0.19658658,  1.71274154, -3.6064863 ],
#     [-3.33214493, -1.40115615, -3.31782413],
#     [ 2.6410079 ,  0.31005138,  2.63615068],
#     [-0.73309096,  1.52406229, -3.83759232],
#     [-3.24622907,  1.89298999, -1.04992091],
#     [-0.64339863,  1.37512122, -4.05727377],
#     [-3.29574477, -0.6436726 ,  1.06367609],
#     [-3.27555403,  1.87367724, -0.89281668],
#     [ 2.79240823,  0.25621821, -2.53066909],
#     [ 1.66965787,  1.74026779, -3.40439946],
#     [-3.04035317,  0.8791836 ,  4.86427347],
#     [-0.77424682,  1.53143229, -3.89310258],
#     [ 2.54846652,  0.71231497, -0.89569754],
#     [-2.69110657,  1.68281874,  2.43489224],
#     [-3.37711015,  1.80146775, -2.6597481 ],
#     [-0.75694007,  1.37998816, -4.01703633],
#     [-0.79293518,  1.39627173, -3.96538257],
#     [ 2.28440808,  2.33897582, -2.3683401 ],
#     [ 2.46440797,  2.32875792,  1.20472059],
#     [ 2.93308087, -1.49040357,  3.76091474],
#     [-0.53848291,  1.48615823, -3.89121629],
#     [ 3.12733995, -1.92840956,  2.4755406 ],
#     [-2.58188077,  1.38387427, -1.60791039],
#     [-0.76196571,  1.40283851, -4.03348653],
#     [ 2.95260969, -1.44937576, -3.7736769 ],
#     [ 2.3420645 ,  0.68554403, -3.99148121],
#     [-0.53024999,  1.42101051, -3.95636371],
#     [-2.51559291,  4.78698116, -0.00594994],
#     [-0.52755258,  1.43412575, -3.95110949],
#     [-2.01559295,  2.03490836,  2.03166186],
#     [ 0.99030766, -0.20983995,  2.36045192],
#     [-2.19089557,  1.25673923, -2.17754248],
#     [-2.70425049, -0.16808945,  4.36861081],
#     [ 2.7658622 ,  0.2318934 ,  2.43641461],
#     [-0.80097931,  1.44099037, -3.92066363],
#     [-1.88588283,  2.16953486, -3.02752834],
#     [ 2.54846652,  1.09601928, -3.81022445],
#     [-3.22909986, -0.95553829, -2.45125471],
#     [-0.63469044,  1.53780985, -3.83170456],
#     [-0.55855883,  1.37921973, -3.98243458],
#     [-0.72697974,  1.53035319, -3.83916153],
#     [-0.52874464,  1.4215586 , -3.94795642],
#     [ 3.10530412,  1.96983083, -2.26624215],
#     [ 2.7358622 , -0.4133808 ,  2.87761284],
#     [-2.43614999,  2.45953187,  0.58525279],
#     [-0.65683392,  1.36028967, -4.04459432],
#     [ 2.6410079 , -0.71139784,  2.19554053]
# ])
# keypoints_3d = np.array([
#     [4.1108, 1.476011, 11.10511],
#     [3.420022, 1.346011, 10.0311],
#     [3.602045, 0.5460111, 8.041292],
#     [5.511046, 0.5460108, 11.48816],
#     [3.408501, 1.566011, 10.76077],
#     [2.958783, 1.337011, 9.475308],
#     [4.301847, 1.346011, 11.8291],
#     [3.80042, 1.768011, 11.96864],
#     [2.561957, 1.763012, 9.679686],
#     [-3.396507, 0.5410113, 16.80474],
#     [-5.334239, 0.6060118, 13.30519],
#     [-3.878393, 1.294012, 13.33021],
#     [-2.5006, 1.321011, 15.64591],
#     [-0.2284485, 1.774012, 16.55072],
#     [0.7360783, 1.766011, 16.00136],
#     [1.771463, 1.754011, 15.40262],
#     [3.045156, 1.786012, 14.59974],
#     [3.959647, 1.756011, 14.10405],
#     [-3.208877, 1.772012, 10.78914],
#     [-2.234848, 1.744012, 10.21599],
#     [-1.198619, 1.741012, 9.624819],
#     [-0.2232526, 1.692011, 9.060117],
#     [0.7499195, 1.731012, 8.444925],
#     [-0.9610124, 7.587012, 10.78686],
#     [1.309767, 4.046011, 14.6413],
#     [0.3029433, 4.016011, 12.60264],
#     [1.346898, 6.216012, 14.66622],
#     [0.1207722, 6.150012, 12.70712],
#     [-0.9741794, 6.186011, 10.53335],
#     [-0.8958945, 4.046011, 10.48895]
# ], dtype=np.float32)

def rotation_vector_to_euler(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0
    return np.degrees([roll, pitch, yaw])

def image_to_world(H, u, v, w):
    H_inv = np.linalg.inv(H)
    image_point = np.array([[u, v, w]], dtype=np.float32).T
    world_point = H_inv @ image_point
    x, y, z = world_point[:4] / world_point[2]  # Normalize
    return x, y, z

def estimate_pose(keypoints_2d, keypoints_3d):
    keypoints_2d = np.array(keypoints_2d)
    print(keypoints_2d)
    _, rotation_vector, translation_vector= cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    # translation, rotation = pose_filter.smooth(translation_vector, rotation_vector)
    return  translation_vector , rotation_vector


def undistort_image(image, K, dist):
    """
    Undistorts the given image using the camera matrix and distortion coefficients.

    Parameters:
    - image (np.ndarray): The distorted input image.
    - K (np.ndarray): The 3x3 camera intrinsic matrix.
    - dist (np.ndarray): The distortion coefficients (1x5).

    Returns:
    - undistorted_image (np.ndarray): The undistorted image.
    """
    # Get image size
    h, w = image.shape[:2]

    # Compute the optimal new camera matrix (optional, can reduce black edges)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)

    # Undistort the image
    undistorted_image = cv2.undistort(image, K, dist, None, new_K)

    # Crop the image to the valid region of interest (optional)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image


class PoseLowPassFilter:
    def __init__(self, alpha=0.05):
        self.prev_pose = None
        self.alpha = alpha  # Smoothing factor (higher = smoother)

    def smooth(self, translation, rotation):
        if self.prev_pose is None:
            self.prev_pose = (translation, rotation)
            return translation, rotation

        smoothed_translation = self.alpha * self.prev_pose[0] + (1 - self.alpha) * translation
        smoothed_rotation = self.alpha * self.prev_pose[1] + (1 - self.alpha) * rotation

        self.prev_pose = (smoothed_translation, smoothed_rotation)
        return smoothed_translation, smoothed_rotation
pose_filter = PoseLowPassFilter(alpha=0.5)

def camera_callback(frame, csv_writer):
    try:
        undistorted_img = undistort_image(frame, camera_matrix, dist_coeffs)
        results = model(undistorted_img, conf=0.75, verbose=False)[0]

        if results.keypoints is None:
            return
        
        keypoints = results.keypoints.xy.cpu().numpy().squeeze()
        
        translation, rotation = estimate_pose(keypoints, keypoints_3d)
        x, y, z = translation.flatten()
        roll, pitch, yaw = rotation_vector_to_euler(rotation)
        rx, ry, rz= image_to_world(H, x, y, z)
        rx = rx.squeeze()
        ry = ry.squeeze()
        rz = rz.squeeze()
        csv_writer.writerow([rx, ry, rz, roll, pitch, yaw])
        print(f"Pose saved: x={x}, y={y}, roll={roll}, pitch={pitch}, yaw={yaw}")

        cv2.putText(frame, f"X: {x:.2f}m, Y: {y:.2f}m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("KeyPoints & Pose", undistorted_img)
        cv2.waitKey(10)

    except Exception as e:
        print(f"Error in processing: {e}")

video_path = "/media/rbccps/Kingston/synthetic_data_testing/diadem_rectangle_trial3.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video stream.")
    exit()

with open("diadem_rectangle3_1.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["X", "Y", "Z", "Roll", "Pitch", "Yaw"])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        camera_callback(frame, csv_writer)

