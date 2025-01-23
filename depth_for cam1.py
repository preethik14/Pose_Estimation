import torch
import cv2
from ZoeDepth.zoedepth.utils.misc import colorize
from PIL import Image
import numpy as np

# Load the ZoeDepth model
zoe = torch.hub.load("/home/rbccps/zero_shot/ZoeDepth", "ZoeD_N", source="local", pretrained=True)
zoe = zoe.to("cuda")

# RTSP Link
rtsp_url = "rtsp://admin:artpark123@192.168.0.220"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

# Generate depth map for each frame
def generate_pointcloud_method2(depth):
    cx = 657.10347183
    cy = 413.00928701
    fx = 429.9493431
    fy = 427.32633934
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    print("Point Cloud Generated")
    return np.dstack((x, y, z))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break

    # Convert the frame (BGR to RGB) and then to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Generate the depth map
    depth = zoe.infer_pil(frame_pil)
    colored_depth = colorize(depth)  # Optional: colorize the depth map

    # Generate point cloud (Optional)
    point_cloud = generate_pointcloud_method2(depth)

    # Display the colored depth map
    colored_depth_bgr = cv2.cvtColor(np.array(Image.fromarray(colored_depth)), cv2.COLOR_RGB2BGR)
    cv2.imshow("Colored Depth", colored_depth_bgr)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
