import argparse
import os
from glob import glob
from os import path

import cv2 as cv
import imageio.v3 as iio
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure, \
    PeakSignalNoiseRatio

# Metrics initialization
ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
psnr_model = PeakSignalNoiseRatio(data_range=1.0).cuda()
lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()

def crop_image(gt_mask, patch_size, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param args: some images with shape of (H, W, C)
    """
    mask_uv = np.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)
    max_v, max_u = mask_uv.max(0)
    pad_size = 0
    min_v = (min_v - pad_size).clip(0, gt_mask.shape[0])
    min_u = (min_u - pad_size).clip(0, gt_mask.shape[1])
    max_v = (max_v + pad_size).clip(0, gt_mask.shape[0])
    max_u = (max_u + pad_size).clip(0, gt_mask.shape[1])
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    for image in args:
        if image is None:
            cropped_images.append(None)
        else:
            cropped_image = np.ones((max_size, max_size, 3), dtype=image.dtype)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, start_u: start_u + len_u] = image[min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[start_v: start_v + len_v, :] = image[min_v: max_v, min_u: max_u]

            cropped_image = cv.resize(cropped_image, (patch_size, patch_size), interpolation=cv.INTER_LINEAR)
            cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]

class MyDataset:
    def __init__(self, gt_paths, render_paths):
        self.gt_paths = gt_paths
        self.render_paths = render_paths

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        im_gt = iio.imread(self.gt_paths[idx])
        im_gsbody = iio.imread(self.render_paths[idx])
        return im_gt, im_gsbody, self.gt_paths[idx]


def evaluate_folder(data_dir):
    # Paths to the files
    filenames = os.listdir(path.join(data_dir, 'gt'))
    filenames.sort()
    gt_paths = [path.join(data_dir, f'gt/{filename}') for filename in filenames]

    render_names = os.listdir(path.join(data_dir, 'result'))
    render_names.sort()
    render_paths = [path.join(data_dir, f'result/{filename}') for filename in render_names]

    # Dataset creation
    dataset = MyDataset(gt_paths, render_paths)
    dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=8, collate_fn=lambda x: x)

    # Metrics list initialisation
    lpips_list = []
    ssim_list = []
    psnr_list = []

    # Image processing
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        im_gt, im_gsbody, datapath = data
        im_gt = torch.tensor(im_gt / 255).permute(2, 0, 1).float().cuda()[None]
        im_gsbody = torch.tensor(im_gsbody / 255).permute(2, 0, 1).float().cuda()[None]


        psnr_value = psnr_model(im_gsbody, im_gt).item()
        ssim_value = ssim_model(im_gsbody, im_gt).item()
        lpips_value = lpips_model(im_gsbody, im_gt).item() #perceptual(im_gsbody, im_gt,  mask=None, use_mask=False).item()

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        lpips_list.append(lpips_value)

    # Averaging metrics
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)

    psnr_val = psnr_list.mean()
    ssim_val = ssim_list.mean()
    lpips_val = lpips_list.mean()

    return psnr_val, ssim_val, lpips_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate image quality metrics')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing gt, render and mask folders')
    args = parser.parse_args()

    #[args.data_dir]
    test_folders = glob(os.path.join(args.data_dir, './'))
    test_folders = sorted(test_folders)
    lpips_list = []
    ssim_list = []
    psnr_list = []

    for folder in test_folders:
        print("Eval" + folder)
        psnr_val, ssim_val, lpips_val = evaluate_folder(folder)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

        print(f"PSNR: {psnr_val}")
        print(f"SSIM: {ssim_val}")
        print(f"LPIPS: {lpips_val}")

    # Averaging metrics
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)

    psnr_val = psnr_list.mean()
    ssim_val = ssim_list.mean()
    lpips_val = lpips_list.mean()

    print(f"Total PSNR: {psnr_val}")
    print(f"Total SSIM: {ssim_val}")
    print(f"Total LPIPS: {lpips_val}")
