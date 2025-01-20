import cv2
import numpy as np

# World points (in meters)
world_points = np.array([[0,0],
                         [0,1.5],
                         [0.5,0.5],
                         [0.5,3.5],
                         [1,1],
                         [1,4.5],
                         [1.5,0.5],
                         [1.5,3],
                         [2,0],
                         [2,1.5],
                         [2,2],
                         [2.5,1],
                        #  [3,1.5],
                        #  [3,2],
                        #  [3.5,0.5],
                        #  [3.5,2.5],
                        #  [3.5,3.5],
                    ], dtype=np.float32)

# Corresponding image points (in pixels)
image_points = np.array([[487,585], 
                         [505,410], 
                         [590,528],
                         [587,275],
                         [682,473],
                         [644,232],
                         [788,534],
                         [724,315],
                         [915,598],
                         [842,435],
                         [829, 394],
                         [940,486],
                        #  [989,448],
                        #  [959,401],
                        #  [1113,545],
                        #  [999,377],
                        #  [955,315]

], dtype=np.float32)

# Camera intrinsic matrix
K = np.array( [[812.16219987,   0,         627.39991698],
 [  0,         795.99847395, 363.71794362],
 [  0,           0,           1.        ]], dtype=np.float32)

# Distortion coefficients
dist = np.array([[-0.54018263,  0.93708365, -0.01391213, -0.05143317, -0.5566984 ]], dtype=np.float32)

# Undistort image points
image_points = image_points.reshape(-1, 1, 2)
undistorted_points = cv2.undistortPoints(image_points, K, dist)

# Convert undistorted points back to pixel coordinates
undistorted_points_pixels = undistorted_points.reshape(-1, 2)

# Compute homography using RANSAC
H, status = cv2.findHomography(world_points, undistorted_points_pixels, cv2.RANSAC, 5.0)
print(H)
# Invert the homography matrix if computing world coordinates from the image coordinates
H_inv = np.linalg.inv(H)
####################################################################################################################################
# Reprojection error Calculation
# Reshape world points for perspective transformation
new_world_points = np.array([ [3,1.5],
                         [3,2],
                         [3.5,0.5],
                         [3.5,2.5],
                         [3.5,3.5]], dtype=np.float32)
new_undistorted_points = cv2.undistortPoints(new_world_points, K, dist)
new_undistorted_points_pixels = new_undistorted_points.reshape(-1, 2)

world_points_reshaped = new_world_points.reshape(-1, 1, 2)

# Transform world points to predicted image points
predicted_image_points = cv2.perspectiveTransform(world_points_reshaped, H)
predicted_image_points_flattened = predicted_image_points.reshape(-1, 2)

# Calculate reprojection error
squared_differences = (predicted_image_points_flattened - new_undistorted_points_pixels) ** 2
sum_squared_differences = np.sum(squared_differences, axis=1)
mse = np.mean(sum_squared_differences)
rmse = np.sqrt(mse)
print(f"Root Mean Square Reprojection error: {rmse:.4f} pixels")
