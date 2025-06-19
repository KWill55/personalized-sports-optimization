import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from mmpose.apis import inference_topdown_pose_model, init_pose_model
from mmpose.structures import merge_data_samples
from mmdet.apis import init_detector, inference_detector

# ========================================
# Config
# ========================================

session = "freethrows1"
det_config = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/' \
                 'faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/' \
                  'hrnet_w32_coco_256x192-6c91a6ed_20200708.pth'

video_path = 'left_freethrow1_sync.mp4.mp4'  # <-- change this
output_csv = 'output_hrnet.csv'

# ========================================
# Landmark Names (COCO order: 17 keypoints)
# ========================================
landmark_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# ========================================
# Load Models
# ========================================

det_model = init_detector(det_config, det_checkpoint, device='cuda:0')
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda:0')

# ========================================
# Process Video
# ========================================

cap = cv2.VideoCapture(video_path)
frame_idx = 0
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run person detection
    det_result = inference_detector(det_model, frame)
    person_bboxes = det_result.pred_instances.bboxes.cpu().numpy()
    person_scores = det_result.pred_instances.scores.cpu().numpy()

    # Use only the top-scoring person
    if len(person_bboxes) == 0:
        keypoints = [[-1, -1, -1] for _ in range(17)]
    else:
        top_idx = np.argmax(person_scores)
        person = dict(bbox=person_bboxes[top_idx])

        pose_result = inference_topdown_pose_model(
            pose_model, frame, [person], format='xyxy', dataset='TopDownCocoDataset', return_vis=False
        )

        keypoints = pose_result[0].pred_instances.keypoints[0].cpu().numpy()
        scores = pose_result[0].pred_instances.keypoint_scores[0].cpu().numpy()

        keypoints = [[x, y, v] for (x, y), v in zip(keypoints, scores)]

    flat = [frame_idx] + [val for pt in keypoints for val in pt]
    results.append(flat)
    frame_idx += 1

cap.release()

# ========================================
# Save CSV
# ========================================
columns = ['frame']
for name in landmark_names:
    columns += [f'{name}_x', f'{name}_y', f'{name}_score']

df = pd.DataFrame(results, columns=columns)
df.to_csv(output_csv, index=False)
print(f"Saved HRNet keypoints to {output_csv}")
