import cv2
import torch
import numpy as np
from PIL import Image
from ALIKE.alike import ALike, configs
from ALIKE.demo import SimpleTracker
from ZoeDepth.zoedepth.utils.misc import colorize

# Initialize ALIKE keypoint detection
alike_model = ALike(**configs['alike-t'], device='cuda', top_k=-1, scores_th=0.2, n_limit=500)
tracker = SimpleTracker()

# Initialize ZoeDepth
zoe = torch.hub.load("/home/rbccps/zero_shot/ZoeDepth", "ZoeD_N", source="local", pretrained=True)
zoe = zoe.to("cuda")

# RTSP Link
rtsp_url = "/home/rbccps/zero_shot/cam2.mp4"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break
    
    # Start total time measurement
    total_start = cv2.getTickCount()

    # ----------------- Keypoint Detection -----------------
    kp_start = cv2.getTickCount()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints_data = alike_model(frame_rgb)
    keypoints = keypoints_data['keypoints']
    descriptors = keypoints_data['descriptors']
    output_frame, matches = tracker.update(frame, keypoints, descriptors)
    kp_duration = (cv2.getTickCount() - kp_start) / cv2.getTickFrequency()

    # ----------------- Depth Estimation -------------------
    depth_start = cv2.getTickCount()
    frame_pil = Image.fromarray(frame_rgb)
    depth_map = zoe.infer_pil(frame_pil)
    colored_depth = colorize(depth_map)
    depth_frame = cv2.cvtColor(np.array(colored_depth), cv2.COLOR_RGB2BGR)
    depth_duration = (cv2.getTickCount() - depth_start) / cv2.getTickFrequency()

    # ----------------- Display Outputs --------------------
    combined_frame = np.hstack((output_frame, depth_frame))
    cv2.imshow("Keypoints and Depth", combined_frame)

    # Measure total time
    total_duration = (cv2.getTickCount() - total_start) / cv2.getTickFrequency()
    print(f"Total Time: {total_duration:.4f}s | Keypoint Time: {kp_duration:.4f}s | Depth Time: {depth_duration:.4f}s")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
