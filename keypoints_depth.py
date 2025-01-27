import cv2
import torch
import numpy as np
from PIL import Image
from ALIKE.alike import ALike, configs
from ALIKE.demo import SimpleTracker
from ZoeDepth.zoedepth.utils.misc import colorize

# Initialize ALIKE keypoint detection
alike_model = ALike(**configs['alike-t'], device='cuda', top_k=-1, scores_th=0.2, n_limit=1000)
tracker = SimpleTracker()

# Initialize ZoeDepth
zoe = torch.hub.load("/home/rbccps/pose/Pose_Estimation/ZoeDepth", "ZoeD_N", source="local", pretrained=True)
zoe = zoe.to("cuda")

# RTSP Link
rtsp_url = "/home/rbccps/pose/Pose_Estimation/cam1_diadem.mp4"
cap = cv2.VideoCapture(rtsp_url)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS:{fps}")
frame_time = 1/fps
window_name = "Keypoints and Depth"

focalLengthX = 812.16219987
focalLengthY = 795.99847395
centerX = 627.39991698
centerY = 363.71794362
scalingFactor = 0.333
def generate_pointcloud_method1(rgb,depth,ply_file):
    
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")


    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            if Z==0: continue
            X = (u - centerX) * Z / focalLengthX
            Y = (v - centerY) * Z / focalLengthX
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))

#     file = open(ply_file,"w")
#     file.write('''ply
# format ascii 1.0
# element vertex %d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property uchar alpha
# end_header
# %s
# '''%(len(points),"".join(points)))
#     file.close()

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
    depth_image = Image.fromarray(depth_map.astype(np.uint16))
    ply_file = "pointcloud.ply"
    generate_pointcloud_method1(frame_pil, depth_image, ply_file)

    depth_frame = cv2.cvtColor(np.array(colored_depth), cv2.COLOR_RGB2BGR)
    depth_duration = (cv2.getTickCount() - depth_start) / cv2.getTickFrequency()

    # ----------------- Display Outputs --------------------
    combined_frame = np.hstack((output_frame, depth_frame))
    cv2.imshow(window_name, combined_frame)
    cv2.resizeWindow(window_name, 600, 300)
    
    # Measure total time
    total_duration = (cv2.getTickCount() - total_start) / cv2.getTickFrequency()
    elapsed_time = total_duration
    wait_time = max(0,frame_time - elapsed_time)
    if wait_time > 0:
        cv2.waitKey(int(wait_time *1000))
    print(f"Total Time: {total_duration:.4f}s | Keypoint Time: {kp_duration:.4f}s | Depth Time: {depth_duration:.4f}s")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
