import cv2
import numpy as np

# World points (in meters)
world_points = np.array([[0,0],
                         [1,1],
                         [2,2],
                         [2,3],
                        [3, 1.5], 
                        [3, 2.5], 
                        [3.5, 2],
                        # [3.5, 0.5],
                        [4, 2], 
                        [4.5, 1], 
                        [4.5, 3],
                        [5, 1.5], 
                        [5, 3.5], 
                        [5.5, 2.5],
                        [6, 1], 
                        [6, 2], 
                        [6, 3.5], 
                        # [6.5, 0.5],
                        # [7, 1.5], 
                        # [7, 2.5], 
                        # [7.5, 2], 
                        # [7.5, 3.5],
                        # [8, 1], 
                        [8.5, 2],
                        [9.5,3]
                    ], dtype=np.float32)

# Corresponding image points (in pixels)
image_points = np.array([[42,440],
                         [56,483],
                         [82,538],
                         [34,596],
    [216, 514], 
    [165,573], 
    [248,546], 
    # [307,466], 
    [308,548], 
    [414,488], 
    [346, 627], 
    [469, 514], 
    [424, 679], 
    [530, 586], 
    [612,483], 
    [614,545], 
    [614,681], 
    # [679,456], 
    # [759,506], 
    # [784,575], 
    # [848,533], 
    # [903,655], 
    # [880,470], 
    [987,519],
    [1172,566]
], dtype=np.float32)

# Camera intrinsic matrix
K = np.array([[429.9493431, 0, 657.10347183],
              [0, 427.32633934, 413.00928701],
              [0, 0, 1]], dtype=np.float32)

# Distortion coefficients
dist = np.array([[0.07899657, -0.07846187, -0.02638088, 0.03126975, 0.01879872]], dtype=np.float32)

# Undistort image points
image_points = image_points.reshape(-1, 1, 2)
undistorted_points = cv2.undistortPoints(image_points, K, dist)

# Convert undistorted points back to pixel coordinates
undistorted_points_pixels = undistorted_points.reshape(-1, 2)

# Compute homography using RANSAC
H, status = cv2.findHomography(world_points, undistorted_points_pixels, cv2.RANSAC, 5.0)
print(H)
# Invert the homography matrix
H_inv = np.linalg.inv(H)
####################################################################################################################################
# Reprojection error Calculation

# Reshape world points for perspective transformation
new_world_points = np.array([[3.5,0.5],[6.5, 0.5],
                        [7, 1.5], 
                        [7, 2.5], [7.5, 2], 
                        [7.5, 3.5],
                        [8, 1]
                        ], dtype=np.float32)
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

