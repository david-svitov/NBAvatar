import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import transformation_util as tranu, system_util
from src.utils import util
from src.utils.train_utils import K_to_camera_matrix


class DecafDataset(Dataset):
    def __init__(self, dataset_path, person_id, cam_ids, mode="train",
                 render_size=1080, sample_each=5, sam3_masks=True):
        self._frames_folder = os.path.join(dataset_path, mode, "frames", person_id)

        camera_list = glob(os.path.join(self._frames_folder, "*"))
        self._camera_list = [cam_id.split('/')[-1] for cam_id in camera_list]
        self._cam_ids = cam_ids
        frames = glob(os.path.join(self._frames_folder, self._camera_list[0], "rgb/*"))
        self._n_frames = len(frames)
        sample = cv2.imread(frames[0])
        self._H, self._W, _ = sample.shape
        self._render_size = render_size
        self._sample_each = sample_each
        self._sam3_masks = sam3_masks

        self._deformations = np.load(os.path.join(dataset_path, mode, "deformations", person_id, "deforms.npy"))
        ignore_idx = np.load(os.path.join(dataset_path, "assets/FLAME_neck_idx.npy"))
        self._deformations[:, ignore_idx] = 0

        params = system_util.pickle_loader(os.path.join(dataset_path, mode, "params", person_id, "params_smoothed.pkl"))
        self._cam_params = system_util.pickle_loader(os.path.join(dataset_path, mode, "cameras", person_id, "cameras.pkl"))
        for cam_id in self._cam_params.keys():
            for key in self._cam_params[cam_id].keys():
                self._cam_params[cam_id][key] = np.array(self._cam_params[cam_id][key])
            self._cam_params[cam_id]["intrinsic_orig"] = self._cam_params[cam_id]["intrinsic"]

        self._parsed_parms = util.params_parser(params)
        self._head_rot_mat = tranu.rot_angle2rotmat(self._parsed_parms['head']['global_orient'])
        self._head_poses = tranu.convert_R_T_to_RT4x4(self._head_rot_mat, self._parsed_parms['head']['transl'])

    def _select_idx(self, data, idx):
        sampled_data = {}
        sampled_data["head"] = {key: value[idx] for key, value in data["head"].items()}
        sampled_data["right_hand"] = {key: value[idx] for key, value in data["right_hand"].items()}
        return sampled_data

    def _crop_and_resize_image(self, image, mask, K):
        crop_size = min(self._W, self._H)
        center_x, center_y = self._W // 2, self._H // 2
        half_size = crop_size // 2
        x, y = center_x - half_size, center_y - half_size
        image = image[y:y + crop_size, x:x + crop_size]
        mask = mask[y:y + crop_size, x:x + crop_size]

        image = cv2.resize(image, (self._render_size, self._render_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self._render_size, self._render_size), interpolation=cv2.INTER_NEAREST)

        scale_factor = self._render_size / crop_size
        K[0, 0] *= scale_factor  # fx
        K[1, 1] *= scale_factor  # fy
        K[0, 2] = (K[0, 2] - x) * scale_factor
        K[1, 2] = (K[1, 2] - y) * scale_factor

        return image, mask, K

    def __len__(self):
        return len(self._cam_ids) * self._n_frames // self._sample_each

    def __getitem__(self, idx):
        idx *= self._sample_each
        cam_num = idx // self._n_frames
        cam_id = self._cam_ids[cam_num]
        frame_id = idx % self._n_frames

        rgb_path = os.path.join(self._frames_folder, cam_id, "rgb", f'{frame_id:06d}.jpg')
        mask_path = os.path.join(self._frames_folder, cam_id, "mask", f'{frame_id:06d}.jpg')
        sam3_path = os.path.join(self._frames_folder, cam_id, "sam3_preproc", f'{frame_id:06d}.jpg')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)[..., ::-1] / 255.
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[..., ::-1] / 255.

        if self._sam3_masks:
            sam3 = cv2.imread(sam3_path, cv2.IMREAD_GRAYSCALE)[..., None]
            sam3 = sam3 / 255.
            mask = mask * sam3

        K = self._cam_params[cam_id]["intrinsic_orig"].copy()
        rgb, mask, K = self._crop_and_resize_image(rgb, mask, K)
        self._cam_params[cam_id]["intrinsic"] = K

        camera_mat = K_to_camera_matrix(K, rgb.shape[1], rgb.shape[0], self._render_size, self._render_size)
        self._cam_params[cam_id]["projection_mat"] = camera_mat

        data = {
            "rgb": torch.from_numpy(rgb).float(),
            "mask": torch.from_numpy(mask).float(),
            "camera": self._cam_params[cam_id],
            "head_rot_mat": self._head_rot_mat[frame_id],
            "head_pose": self._head_poses[frame_id],
            "model_params": self._select_idx(self._parsed_parms, frame_id),
            "deformation": self._deformations[frame_id],
            "pose_idx": frame_id,
            "cam_idx": cam_id,
        }
        return data
