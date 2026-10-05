import argparse
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def get_image(cap, cap_seg=None, frame_id=None):
    # ret, img = cap.read()
    # get image specified by frame_id

    if frame_id is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, img = cap.read()

    '''
    img = cv2.putText( img,#cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                          str(frame_id),(100,100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          4, 
                          (0, 255, 0), 
                          2, 
                          cv2.LINE_AA)
    '''
    # img = np.concatenate((img,seg_img),axis=1)
    if cap_seg is not None:
        if frame_id is not None:
            cap_seg.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, seg_img = cap_seg.read()
        # seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        # seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
        return img, seg_img  # np.array(img),np.array(seg_img)
    else:
        return np.array(img), None


def process_video(vid_path:str, seg_path:str, result_path:str)->None:
    # load masking image and RGB image
    print("START Processing video:", vid_path)
    cap = cv2.VideoCapture(vid_path)  
    seg_cap = cv2.VideoCapture(seg_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rgb_save_folder = os.path.join(result_path, "rgb")
    mask_save_folder = os.path.join(result_path, "mask")
    if not os.path.exists(rgb_save_folder):
        os.makedirs(rgb_save_folder)
    if not os.path.exists(mask_save_folder):
        os.makedirs(mask_save_folder)

    for i in tqdm(range(0, n_frames)):
        image, mask = get_image(cap, seg_cap)

        # mask the image
        #ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        #mask[mask!=0] = 1

        cv2.imwrite(os.path.join(rgb_save_folder, f'{i:06d}.jpg'), image)
        cv2.imwrite(os.path.join(mask_save_folder, f'{i:06d}.jpg'), mask)

    cap.release()
    seg_cap.release()
    print("FINISH Processing video:", vid_path)

def process_video_wrapper(args_tuple):
    vid_path, seg_path, result_path = args_tuple
    process_video(vid_path, seg_path, result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuratoins') 
    parser.add_argument('--dataset_path', type=str, default="../DecafDataset/")
    args = parser.parse_args()
    # set path to the RGB vieo and segmentation video in DecafDataset
    split = "test"
    video_path = os.path.join(args.dataset_path, split, 'videos/S2/*')
    print(video_path)
    videos_list = glob(video_path)

    #tasks = []
    for vid_path in videos_list:
        id_folder = vid_path.split('/')[-2]
        video_name = vid_path.split('/')[-1]
        seg_path = os.path.join(args.dataset_path, split, "segmentations", id_folder, video_name)
        result_path = os.path.join(args.dataset_path, split, "frames", id_folder, video_name.split('.mp4')[0])
        #tasks.append((vid_path, seg_path, result_path))
        process_video(vid_path, seg_path, result_path)

    #with multiprocessing.Pool(processes=8) as pool:
    #    pool.map(process_video_wrapper, tasks)
