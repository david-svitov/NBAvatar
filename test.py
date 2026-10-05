import argparse
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import statistics

from src.utils.train_utils import dict2device
from src.utils.util import instantiate_from_config


def make_compare(gt_np, render_np, psnr: float) -> np.ndarray:
    """Return a uint8 BGR image: GT | render | error map with PSNR label.

    Args:
        gt_np:     (H, W, 3) float32 RGB in [0, 1]
        render_np: (H, W, 3) float32 RGB in [0, 1]
        psnr:      scalar PSNR value
    """
    gt_bgr     = (gt_np    [..., ::-1] * 255).astype(np.uint8)
    render_bgr = (render_np[..., ::-1] * 255).astype(np.uint8)

    err = np.abs(render_np - gt_np).max(axis=-1)          # (H, W) in [0, 1]
    err_u8  = (err * 255).astype(np.uint8)
    err_map = cv2.applyColorMap(err_u8, cv2.COLORMAP_JET) # (H, W, 3) BGR

    comparison = np.concatenate([gt_bgr, render_bgr, err_map], axis=1).copy()

    h = comparison.shape[0]
    cv2.putText(comparison, 'GT | render | error',
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(comparison, f'PSNR {psnr:.2f} dB',
                (10, h - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return comparison


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="Paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="Load pretrained weights from the checkpoint",
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()

    opt, unknown = parser.parse_known_args(sys.argv)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    trainer = instantiate_from_config(config.trainer, additional_params={"test_mode": True})
    trainer.to(trainer.device)
    checkpoint_path = opt.pretrained
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # for old checkpoints
    checkpoint['model_state_dict'] = {
        key: value for key, value in checkpoint['model_state_dict'].items()
        if not key.startswith('_renderer_dumb.')
    }

    trainer.load_state_dict(checkpoint['model_state_dict'])

    dataset = instantiate_from_config(config.dataset_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    exp_name = opt.base[0].split('/')[-1].split('.')[0]
    chekpoint_name = "test-" + checkpoint_path.split('/')[-3]
    log_folder_name = os.path.join(config.log_dir, exp_name, chekpoint_name)
    os.makedirs(log_folder_name, exist_ok=True)

    print("Save to:", log_folder_name)
    gt_folder = os.path.join(log_folder_name, "gt")
    os.makedirs(gt_folder, exist_ok=True)
    predict_folder = os.path.join(log_folder_name, "result")
    os.makedirs(predict_folder, exist_ok=True)
    mask_folder = os.path.join(log_folder_name, "mask")
    os.makedirs(mask_folder, exist_ok=True)
    compare_folder = os.path.join(log_folder_name, "compare")
    os.makedirs(compare_folder, exist_ok=True)

    fps_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = dict2device(batch, trainer.device)
            b, h, w, c = batch["rgb"].shape
            white_backgound = torch.ones(b, h, w, 3).to(trainer.device)
            if trainer.use_renderer:
                batch["background"] = torch.full((b, h, w, trainer.num_channels), 0.5).to(trainer.device)
            else:
                # No renderer to compensate for an arbitrary internal background,
                # so rasterize straight onto the color we want in the final image.
                batch["background"] = white_backgound

            with torch.no_grad(), torch.inference_mode():
                batch.update(trainer.render(batch, test_mode=True))

            fps = 1000 / batch["run_time"]
            fps_list.append(fps)

            #render = batch["render_feautures"][..., 3:]
            if trainer.use_renderer:
                render = batch["render_rgb"] * batch["render_mask"] + white_backgound * (1 - batch["render_mask"])
            else:
                render = batch["render_rgb"]
            gt = batch["rgb"] * batch["mask"] + white_backgound * (1 - batch["mask"])

            render_np = render.cpu().detach().numpy()[0]   # (H, W, 3) float32
            gt_np     = gt.cpu().detach().numpy()[0]        # (H, W, 3) float32

            cv2.imwrite(os.path.join(predict_folder, f"{i:05d}.png"), render_np[..., ::-1] * 255)
            cv2.imwrite(os.path.join(gt_folder,      f"{i:05d}.png"), gt_np[..., ::-1] * 255)

            log_image = batch["billboards_alpha"].cpu().detach().numpy()[0]
            cv2.imwrite(os.path.join(mask_folder, f"{i:05d}.png"), log_image[..., ::-1] * 255)

            mse  = float(((render_np - gt_np) ** 2).mean())
            psnr = -10.0 * np.log10(mse + 1e-8)
            cv2.imwrite(os.path.join(compare_folder, f"{i:05d}.jpg"),
                        make_compare(gt_np, render_np, psnr))

    print("FPS:", statistics.mean(fps_list[10:]))