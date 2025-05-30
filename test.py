import cv2
import numpy as np
from ultralytics import YOLO  # pip install ultralytics

# Load YOLOv8 model
model = YOLO("/media/rbccps/HDD/cad/runs/pose/train6/weights/best.pt")  # replace with your custom model if needed

# Camera intrinsics (replace with actual calibrated values)
fx, fy = 800, 800
cx, cy = 320, 240
camera_matrix = np.array([[1407.40701,   -4.29824404,  948.040273],
 [   0.0,       1402.06664,   522.432758],
 [   0.0,          0.0,         1.0]])
dist_coeffs = np.array([[ 0.04826876],
 [-1.3534217 ],
 [ 4.74603072],
 [-5.47635312]]
, dtype=np.float32)
# Define 3D keypoints in object coordinate system (e.g. corners of square marker)
object_points = np.array([
    [-0.05, 0.05, 0],   # top-left
    [ 0.05, 0.05, 0],   # top-right
    [ 0.05, -0.05, 0],  # bottom-right
    [-0.05, -0.05, 0],  # bottom-left
], dtype=np.float32)

# Define axis for visualization (in meters)
axis_points = np.float32([
    [0, 0, 0],       # origin
    [0.05, 0, 0],    # X-axis endpoint
    [0, 0.05, 0],    # Y-axis endpoint
    [0, 0, 0.05],    # Z-axis endpoint
])


# Open video
cap = cv2.VideoCapture("/media/rbccps/Kingston/synthetic_data_testing/diadem_rectangle_trial3.mp4")  # replace with your video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].astype(int)

        # Define 2D keypoints manually from bounding box
        image_points = np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ], dtype=np.float32)

        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                           camera_matrix, dist_coeffs)

        if success:
            # Project axis for visualization
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int).reshape(-1, 2)

            origin = tuple(imgpts[0])
            frame = cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 2)  # X-axis (red)
            frame = cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 2)  # Y-axis (green)
            frame = cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 2)  # Z-axis (blue)
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("YOLO + Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
