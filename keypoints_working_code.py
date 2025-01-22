import cv2
import logging
import argparse
import numpy as np
from ALIKE.alike import ALike, configs
from ALIKE.demo import ImageLoader, SimpleTracker

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ALike Demo.')
    parser.add_argument('input', type=str, help='RTSP link or video file path or camera input')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l'], default="alike-t", help="The model configuration")
    parser.add_argument('--device', type=str, default='cuda', help="Running device (default: cuda).")
    parser.add_argument('--top_k', type=int, default=-1, help='Detect top K keypoints.')
    parser.add_argument('--scores_th', type=float, default=0.2, help='Detector score threshold.')
    parser.add_argument('--n_limit', type=int, default=5000, help='Maximum number of keypoints.')
    parser.add_argument('--no_display', action='store_true', help='Do not display images to screen.')
    parser.add_argument('--no_sub_pixel', action='store_true', help='Do not detect sub-pixel keypoints.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize image loader and model
    image_loader = ImageLoader(args.input)
    model = ALike(**configs[args.model], device=args.device, top_k=args.top_k, scores_th=args.scores_th, n_limit=args.n_limit)
    tracker = SimpleTracker()

    # Process stream
    if not args.no_display:
        logging.info("Press 'q' to stop!")
        cv2.namedWindow(args.model)

    runtime = []
    for img in image_loader:
        if img is None:
            break
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = model(img_rgb, sub_pixel=not args.no_sub_pixel)
        kpts = pred['keypoints']
        desc = pred['descriptors']
        runtime.append(pred['time'])

        out, N_matches = tracker.update(img, kpts, desc)

        ave_fps = (1. / np.stack(runtime)).mean()
        status = f"Fps:{ave_fps:.1f}, Keypoints/Matches: {len(kpts)}/{N_matches}"
        print(kpts)
        print(desc)
        if not args.no_display:
            cv2.setWindowTitle(args.model, args.model + ': ' + status)
            cv2.imshow(args.model, out)
            if cv2.waitKey(1) == ord('q'):
                break

    logging.info('Finished!')
    if not args.no_display:
        logging.info('Press any key to exit!')
        cv2.waitKey()

if __name__ == "__main__":
    main()
