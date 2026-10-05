import math

import numpy as np
import torch
import torch.nn.functional as F


def dict2device(data_dict, device):
    if hasattr(data_dict, 'to'):
        return data_dict.to(device).float()

    if type(data_dict) == dict:
        for k, v in data_dict.items():
            data_dict[k] = dict2device(v, device)
    return data_dict


def precalc_points(points, camera_mat, camera_transform):
    device = points.device

    homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                             dtype=points.dtype,
                             device=device)
    # Convert the points to homogeneous coordinates
    points_h = torch.cat([points, homog_coord], dim=-1)

    projected_points = torch.einsum('bki,bji->bjk', [camera_transform, points_h])
    projected_points = torch.einsum('bki,bji->bjk', [camera_mat, projected_points])
    return projected_points


def K_to_camera_matrix(K, src_W, src_H, render_W, render_H):
    fy_src = K[1, 1]
    fov_y = 2 * math.atan(src_H / (2 * fy_src))
    fy_render = render_H / (2 * math.tan(fov_y / 2))

    fx_src = K[0, 0]
    fx_render = fx_src * (fy_render / fy_src)

    scale_x = render_W / src_W
    scale_y = render_H / src_H
    cx_render = K[0, 2] * scale_x
    cy_render = K[1, 2] * scale_y

    P = np.zeros((4, 4))

    P[0, 0] = 2.0 * fx_render / render_W
    P[0, 2] = 2.0 * (cx_render / render_W) - 1.0

    P[1, 1] = 2.0 * fy_render / render_H
    P[1, 2] = 2.0 * (cy_render / render_H) - 1.0

    znear, zfar = 0.01, 100.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -zfar * znear / (zfar - znear)
    P[3, 2] = 1.0

    return P


def shift_image(image, shift_x_pixels, shift_y_pixels):
    shift_x = int(round(shift_x_pixels))
    shift_y = int(round(shift_y_pixels))

    shifted = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))

    return shifted

def shift_image_pad(image, shift_x_pixels, shift_y_pixels):
    shift_x = int(round(shift_x_pixels))
    shift_y = int(round(shift_y_pixels))

    pad_left = max(shift_x, 0)
    pad_right = max(-shift_x, 0)
    pad_top = max(shift_y, 0)
    pad_bot = max(-shift_y, 0)

    padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bot))
    H, W = image.shape[-2:]

    return padded[..., pad_top:pad_top + H, pad_left:pad_left + W]
