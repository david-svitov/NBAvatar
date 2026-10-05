import argparse
import sys

import cv2
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

import src.utils.open3d_util as o3du
from src.utils.train_utils import dict2device
from src.utils.util import instantiate_from_config


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
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Flip the billboard rasterization horizontally before the renderer",
    )
    return parser

def rotate_camera_around_center(extrinsic_tensor, center, axis='y', angle_degrees=0):
    device = extrinsic_tensor.device
    dtype = extrinsic_tensor.dtype

    angle_rad = torch.deg2rad(torch.tensor(angle_degrees, device=device, dtype=dtype))

    if axis == 'x':
        R = torch.eye(3, device=device, dtype=dtype)
        R[1, 1] = torch.cos(angle_rad)
        R[1, 2] = -torch.sin(angle_rad)
        R[2, 1] = torch.sin(angle_rad)
        R[2, 2] = torch.cos(angle_rad)
    elif axis == 'y':
        R = torch.eye(3, device=device, dtype=dtype)
        R[0, 0] = torch.cos(angle_rad)
        R[0, 2] = torch.sin(angle_rad)
        R[2, 0] = -torch.sin(angle_rad)
        R[2, 2] = torch.cos(angle_rad)
    elif axis == 'z':
        R = torch.eye(3, device=device, dtype=dtype)
        R[0, 0] = torch.cos(angle_rad)
        R[0, 1] = -torch.sin(angle_rad)
        R[1, 0] = torch.sin(angle_rad)
        R[1, 1] = torch.cos(angle_rad)
    else:
        raise ValueError("axis should be 'x', 'y' or 'z'")

    R_cam = extrinsic_tensor[0, :3, :3]
    t_cam = extrinsic_tensor[0, :3, 3]

    position_relative_to_center = t_cam - center
    rotated_position = torch.matmul(R, position_relative_to_center)
    new_camera_position = rotated_position + center
    new_R_cam = torch.matmul(R_cam, R.t())

    new_extrinsic = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    new_extrinsic[0, :3, :3] = new_R_cam
    new_extrinsic[0, :3, 3] = new_camera_position

    return new_extrinsic


def get_camera_look_at_point(extrinsic, distance=1.0):
    camera_pos = extrinsic[0, :3, 3]

    look_dir_camera = torch.tensor([0, 0, -1], device=extrinsic.device,
                                   dtype=extrinsic.dtype)
    look_dir_world = torch.matmul(extrinsic[0, :3, :3], look_dir_camera)
    look_dir_world = look_dir_world / torch.norm(look_dir_world)

    look_at_point = camera_pos + look_dir_world * distance

    return look_at_point

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

    '''
    # Setup trainable alpha textures
    npoints = len(trainer._texture_alpha)
    texture_alpha = trainer._texture_alpha_init.repeat(npoints, 1, 1)
    texture_alpha = trainer.inverse_opacity_activation(texture_alpha)
    trainer._texture_alpha = torch.nn.Parameter(texture_alpha)

    # Setup trainable RGB textures
    texture_size = texture_alpha.shape[-1]
    base_colors = torch.rand([npoints, trainer.num_channels, 1, 1])
    texture_color = base_colors.expand(-1, -1, texture_size, texture_size)
    #texture_color = torch.ones([npoints, trainer.num_channels, texture_size, texture_size]) * 0.1
    texture_color = texture_color.type(torch.float)
    texture_color = trainer.inverse_color_activation(texture_color.to("cuda"))
    trainer._texture_color = torch.nn.Parameter(texture_color)
    '''

    dataset = instantiate_from_config(config.dataset_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    batch = dict2device(batch, trainer.device)

    rotation_speed = 2.0
    current_extrinsic = batch["camera"]["extrinsic"].clone()
    center = get_camera_look_at_point(current_extrinsic, 0.1)

    vis, cam = o3du.view_setup_known_cam(ext=np.eye(4))
    frame_num = 0
    while True:
        '''
        try:
            batch = next(dataloader_iter)
            batch = dict2device(batch, device)
        except StopIteration:
            batch = next(dataloader_iter)
        '''

        batch["camera"]["extrinsic"] = current_extrinsic

        b, h, w, c = batch["rgb"].shape
        batch["background"] = torch.ones(b, h, w, 6).to(trainer.device)
        white_backgound = torch.ones(b, h, w, 3).to(trainer.device)
        batch.update(trainer.render(batch, test_mode=True, flip=opt.flip))
        render = batch["render_rgb"] * batch["render_mask"] + white_backgound * (1 - batch["render_mask"])
        #render = batch["render_feautures"][..., 3:] * batch["billboards_alpha"] + white_backgound * (1 - batch["billboards_alpha"])
        #render = batch["billboards_alpha"]
        #render = batch["mask"]

        head_mesh = o3du.get_mesh_from_vs_and_faces(batch["deformed_head_vs_in_cam"].cpu().numpy()[0], trainer._flame_faces.cpu().numpy())
        rh_mesh = o3du.get_mesh_from_vs_and_faces(batch["rh_vs_in_cam"].cpu().numpy()[0], trainer._rh_faces.cpu().numpy())
        o3du.visualization_process_o3d(vis, [head_mesh, rh_mesh], cam)

        log_image = render.cpu().detach().numpy()[0]
        cv2.imshow("billbords", log_image[..., ::-1])
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('d'):
            current_extrinsic = rotate_camera_around_center(
                current_extrinsic, center, 'x', rotation_speed
            )
        elif key == ord('a'):
            current_extrinsic = rotate_camera_around_center(
                current_extrinsic, center, 'x', -rotation_speed
            )
        elif key == ord('q'):
            current_extrinsic = rotate_camera_around_center(
                current_extrinsic, center, 'y', rotation_speed
            )
        elif key == ord('e'):
            current_extrinsic = rotate_camera_around_center(
                current_extrinsic, center, 'y', -rotation_speed
            )
        elif key == ord('w'):
            current_extrinsic = rotate_camera_around_center(
                current_extrinsic, center, 'z', rotation_speed
            )
        elif key == ord('s'):
            current_extrinsic = rotate_camera_around_center(
                current_extrinsic, center, 'z', -rotation_speed
            )
        elif key == ord('p'):
            frame_num += 1
            print("Frame:", frame_num)
            batch = next(dataloader_iter)
            batch = dict2device(batch, trainer.device)
