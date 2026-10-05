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
                 render_size=1080, sample_each=1, sam3_masks=True):
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
        self._cam_crop_start = {}
        self._cam_crop_size = {}
        other_mode = "test" if mode == "train" else "train"
        for cam_id in self._cam_params.keys():
            for key in self._cam_params[cam_id].keys():
                self._cam_params[cam_id][key] = np.array(self._cam_params[cam_id][key])

            rh_path = hd_path = None
            for m in (mode, other_mode):
                rh = os.path.join(dataset_path, m, 'right_hand_bbs', person_id, f'{cam_id}.npy')
                hd = os.path.join(dataset_path, m, 'head_bbs', person_id, f'{cam_id}.npy')
                if os.path.exists(rh) and os.path.exists(hd):
                    rh_path, hd_path = rh, hd
                    break

            if rh_path is not None:
                rh_bb = np.rint(np.load(rh_path)).astype(int)
                head_bb = np.rint(self.enlarge_bbox(np.load(hd_path), scale_factor=1.2)).astype(int)
                self._cam_crop_start[cam_id], self._cam_crop_size[cam_id] = \
                    self.find_square_bbox(np.concatenate([rh_bb, head_bb], 0), self._H, self._W)
            else:
                crop_size = min(self._H, self._W)
                self._cam_crop_start[cam_id] = (self._W // 2 - crop_size // 2, self._H // 2 - crop_size // 2)
                self._cam_crop_size[cam_id] = crop_size

            x_min, y_min = self._cam_crop_start[cam_id]
            self._cam_params[cam_id]["intrinsic"][0, 2] -= x_min
            self._cam_params[cam_id]["intrinsic"][1, 2] -= y_min
            self._cam_params[cam_id]["intrinsic"][0:2, :] *= render_size / self._cam_crop_size[cam_id]

        self._parsed_parms = util.params_parser(params)
        self._head_rot_mat = tranu.rot_angle2rotmat(self._parsed_parms['head']['global_orient'])
        self._head_poses = tranu.convert_R_T_to_RT4x4(self._head_rot_mat, self._parsed_parms['head']['transl'])

    def enlarge_bbox(self, bboxes, scale_factor=1.2):
        bboxes = np.array(bboxes, dtype=float)
        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]

        new_widths = widths * scale_factor
        new_heights = heights * scale_factor

        new_bboxes = np.zeros_like(bboxes)
        new_bboxes[:, 0] = centers[:, 0] - new_widths / 2  # xmin
        new_bboxes[:, 1] = centers[:, 1] - new_heights / 2  # ymin
        new_bboxes[:, 2] = centers[:, 0] + new_widths / 2  # xmax
        new_bboxes[:, 3] = centers[:, 1] + new_heights / 2  # ymax

        return new_bboxes

    def find_square_bbox(self, bboxes: np.ndarray, image_height: int, image_width: int):
        """
        Find the greatest square bounding box that contains all bboxes and fits within the image.

        Args:
            bboxes: numpy array of shape (N, 4) where each row is [xmin, ymin, xmax, ymax]
            image_height: height of the image
            image_width: width of the image

        Returns:
            tuple: (min_x, min_y, side_length) of the square bounding box
        """
        # discard bb with non positive values
        indices = np.where(~(bboxes <= 0).any(axis=1))[0]
        bboxes = bboxes[indices]

        # Find min and max ranges across all frames
        all_xmin = bboxes[:, 0].min()
        all_ymin = bboxes[:, 1].min()
        all_xmax = bboxes[:, 2].max()
        all_ymax = bboxes[:, 3].max()

        # Calculate width and height of the bounding region
        width = all_xmax - all_xmin
        height = all_ymax - all_ymin

        # Use the longest side to make it square
        side_length = max(width, height)
        side_length = min(side_length, image_height)

        # Calculate the center of the original bbox
        center_x = (all_xmin + all_xmax) / 2
        center_y = (all_ymin + all_ymax) / 2

        # Calculate initial square coordinates
        min_x = center_x - side_length / 2
        min_y = center_y - side_length / 2

        # Adjust if box goes outside image boundaries
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if min_x + side_length > image_width:
            min_x = image_width - side_length
        if min_y + side_length > image_height:
            min_y = image_height - side_length

        return (int(min_x), int(min_y)), int(side_length)

    def _select_idx(self, data, idx):
        sampled_data = {}
        sampled_data["head"] = {key: value[idx] for key, value in data["head"].items()}
        sampled_data["right_hand"] = {key: value[idx] for key, value in data["right_hand"].items()}
        return sampled_data

    def __len__(self):
        return self._n_frames * len(self._cam_ids) // self._sample_each

    def __getitem__(self, idx):
        cam_idx = idx % len(self._cam_ids)
        frame_id = (idx // len(self._cam_ids)) * self._sample_each
        cam_id = self._cam_ids[cam_idx]

        rgb_path = os.path.join(self._frames_folder, cam_id, "rgb", f'{frame_id:06d}.jpg')
        mask_path = os.path.join(self._frames_folder, cam_id, "mask", f'{frame_id:06d}.jpg')
        sam3_path = os.path.join(self._frames_folder, cam_id, "sam3_preproc", f'{frame_id:06d}.jpg')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)[..., ::-1] / 255.
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_gray > 127).astype(np.float32)
        mask = np.stack([mask, mask, mask], axis=-1)

        if self._sam3_masks and os.path.exists(sam3_path):
            sam3 = cv2.imread(sam3_path, cv2.IMREAD_GRAYSCALE)[..., None].astype(np.float32) / 255.
            mask = mask * sam3

        K = self._cam_params[cam_id]["intrinsic"].copy()
        xmin, ymin = self._cam_crop_start[cam_id]
        crop_size = self._cam_crop_size[cam_id]
        rgb = cv2.resize(rgb[ymin:ymin + crop_size, xmin:xmin + crop_size, :], (self._render_size, self._render_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask[ymin:ymin + crop_size, xmin:xmin + crop_size, :], (self._render_size, self._render_size), interpolation=cv2.INTER_NEAREST)

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
