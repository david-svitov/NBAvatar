import argparse
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from tqdm import tqdm

def union_top3(output, thr=0.5):
    masks = output["masks"]
    k = min(3, len(masks))
    total = None

    for i in range(k):
        m = masks[i][0].cpu().detach().numpy()
        m = (m > thr).astype(np.uint8)

        if total is None:
            total = m
        else:
            total = total | m

    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to data to process")
    args = parser.parse_args()
    dataset_path = args.dataset_path

    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    frames = glob(os.path.join(dataset_path, "*/rgb/*.jpg"))
    frames_processed = glob(os.path.join(dataset_path, "*/sam3_fixed_k/*.jpg"))

    for path in frames_processed:
        path = path.replace("sam3_fixed_k", "rgb")
        if path in frames:
            idx = frames.index(path)
            del frames[idx]

    for image_path in tqdm(frames):
        file_name = image_path.split('/')[-1]
        folder_name = os.path.join(dataset_path, image_path.split('/')[-3], "sam3_fixed_k/")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        image = Image.open(image_path)
        inference_state = processor.set_image(image)

        total_mask = None
        for promt in ["head and neck", "head or neck or torso skin", "hand"]:
            output = processor.set_text_prompt(state=inference_state, prompt=promt)
            if len(output["masks"]) <= 0:
                local_mask = np.zeros_like(image)[..., 0]
            else:
                if promt == "hand":
                    mask = output["masks"][0][0].cpu().detach().numpy()
                else:
                    mask = union_top3(output)
                local_mask = mask

            if total_mask is None:
                total_mask = local_mask
            else:
                total_mask = total_mask | local_mask

        cv2.imwrite(folder_name + file_name, total_mask.astype(np.float32) * 255)
