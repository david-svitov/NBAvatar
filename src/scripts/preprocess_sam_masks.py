import argparse
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to data to process")
    args = parser.parse_args()
    dataset_path = args.dataset_path

    frames = glob(os.path.join(dataset_path, "*/sam3_fixed_k/*.jpg"))

    for image_path in tqdm(frames):
        file_name = image_path.split('/')[-1]
        folder_name = os.path.join(dataset_path, image_path.split('/')[-3], "sam3_preproc/")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        sam3 = cv2.imread(image_path)
        _, sam3 = cv2.threshold(sam3, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((10, 10), np.uint8)
        sam3 = cv2.morphologyEx(sam3, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(folder_name + file_name, sam3.astype(np.float32))
