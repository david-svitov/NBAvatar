import argparse
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from tqdm import tqdm


def enlarge_bbox(bbox_xyxy, scale=1.2):
    """Scale bbox by `scale` around its center, keeping float coords."""
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def is_valid_bbox(bbox_xyxy):
    return (bbox_xyxy > 0).all() and (bbox_xyxy[2] > bbox_xyxy[0]) and (bbox_xyxy[3] > bbox_xyxy[1])


def xyxy_to_norm_cxcywh(bbox_xyxy, img_w, img_h):
    """Convert [xmin, ymin, xmax, ymax] pixel bbox to normalized [cx, cy, w, h]."""
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0 / img_w
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0 / img_h
    w = (bbox_xyxy[2] - bbox_xyxy[0]) / img_w
    h = (bbox_xyxy[3] - bbox_xyxy[1]) / img_h
    return [cx, cy, w, h]


def union_masks(output, thr=0.5):
    masks = output["masks"]
    total = None
    for mask in masks:
        m = mask[0].cpu().detach().numpy()
        m = (m > thr).astype(np.uint8)
        total = m if total is None else (total | m)
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--mode", type=str, default="train", help="Dataset split: train or test")
    parser.add_argument("--person_id", type=str, required=True, help="Person ID, e.g. S3")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    mode = args.mode
    person_id = args.person_id

    frames_root = os.path.join(dataset_path, mode, "frames", person_id)
    rh_bbs_root = os.path.join(dataset_path, mode, "right_hand_bbs", person_id)
    head_bbs_root = os.path.join(dataset_path, mode, "head_bbs", person_id)

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    cam_dirs = sorted(glob(os.path.join(frames_root, "*")))
    cam_ids = [os.path.basename(d) for d in cam_dirs]

    for cam_id in cam_ids:
        rh_bb_path = os.path.join(rh_bbs_root, f"{cam_id}.npy")
        head_bb_path = os.path.join(head_bbs_root, f"{cam_id}.npy")

        has_rh_bbs = os.path.exists(rh_bb_path)
        has_head_bbs = os.path.exists(head_bb_path)

        rh_bbs = np.rint(np.load(rh_bb_path)).astype(int) if has_rh_bbs else None
        head_bbs = np.rint(np.load(head_bb_path)).astype(int) if has_head_bbs else None

        frames = sorted(glob(os.path.join(frames_root, cam_id, "rgb", "*.jpg")))

        out_dir = os.path.join(frames_root, cam_id, "sam3_boxes")
        os.makedirs(out_dir, exist_ok=True)

        for image_path in tqdm(frames, desc=f"{person_id}/{cam_id}"):
            file_name = os.path.basename(image_path)
            out_path = os.path.join(out_dir, file_name)

            if os.path.exists(out_path):
                continue

            frame_id = int(os.path.splitext(file_name)[0])

            image = Image.open(image_path)
            img_w, img_h = image.size

            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)

            # Set semantic text prompt first so add_geometric_prompt uses it instead of
            # the dummy "visual" fallback. Text says WHAT to find; boxes say WHERE.
            inference_state = processor.set_text_prompt(state=inference_state, prompt="skin")

            has_any_prompt = False

            if has_rh_bbs and frame_id < len(rh_bbs) and is_valid_bbox(rh_bbs[frame_id]):
                box = xyxy_to_norm_cxcywh(enlarge_bbox(rh_bbs[frame_id]), img_w, img_h)
                inference_state = processor.add_geometric_prompt(box=box, label=True, state=inference_state)
                has_any_prompt = True

            if has_head_bbs and frame_id < len(head_bbs) and is_valid_bbox(head_bbs[frame_id]):
                box = xyxy_to_norm_cxcywh(enlarge_bbox(head_bbs[frame_id]), img_w, img_h)
                inference_state = processor.add_geometric_prompt(box=box, label=True, state=inference_state)
                has_any_prompt = True

            if not has_any_prompt or len(inference_state.get("masks", [])) == 0:
                total_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            else:
                total_mask = union_masks(inference_state)

            # Draw enlarged bboxes for debug: hand=green, head=blue
            debug_img = cv2.cvtColor((total_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if has_rh_bbs and frame_id < len(rh_bbs) and is_valid_bbox(rh_bbs[frame_id]):
                x1, y1, x2, y2 = enlarge_bbox(rh_bbs[frame_id]).astype(int)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if has_head_bbs and frame_id < len(head_bbs) and is_valid_bbox(head_bbs[frame_id]):
                x1, y1, x2, y2 = enlarge_bbox(head_bbs[frame_id]).astype(int)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            print(out_path)
            cv2.imwrite(out_path, debug_img)
