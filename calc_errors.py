'''This file has been used for the research project course (ENGN8601/ENGN8602) by Namas Bhandari of the Australian National University. The file can be found on github.com/namas191297/evaluating_mvs_in_cpc'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torch.nn import functional as F
import torch

def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())

def AbsDepthError_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())


epes = []
px1s = []
px3s = []

ref_dir = 'ref_images'

print('Calculating metrics...')

for idx, im in enumerate(os.listdir(ref_dir)):
    if 'image' in im:
        curr_dir = ref_dir + '/' + im
        est_mask = np.load(curr_dir + '/' + 'est_mask.npy')
        gt_dp = np.load(curr_dir + '/' + 'gt_depthmap.npy')[0][0]
        est_dp = np.load(curr_dir + '/' + 'est_depthmap.npy')

        est_dp = torch.from_numpy(est_dp)
        est_dp = F.interpolate(est_dp, scale_factor=2, mode="bilinear")[0][0]
        gt_dp = torch.from_numpy(gt_dp)
        
        epes.append(AbsDepthError_metrics(est_dp, gt_dp, est_mask))
        px1s.append(Thres_metrics(est_dp, gt_dp, est_mask, 1))
        px3s.append(Thres_metrics(est_dp, gt_dp, est_mask, 3))
        print(f'Image {idx+1}/{len(os.listdir(ref_dir))}... DONE')

print(f'EPE:{np.mean(epes)}, 1PX:{np.mean(px1s)}, 3PX:{np.mean(px3s)}')

