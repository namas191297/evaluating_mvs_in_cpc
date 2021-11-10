'''This file has been used for the research project course (ENGN8601/ENGN8602) by Namas Bhandari of the Australian National University. The file can be found on github.com/namas191297/evaluating_mvs_in_cpc'''

import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
from PIL import Image
import numpy as np
import re
from plyfile import PlyElement, PlyData
from sklearn import linear_model
import os


imgs_dir = 'ref_images'
fused_scene_dir = 'final_fused_scenes/'

if os.path.exists(fused_scene_dir):
    pass
else:
    os.mkdir(fused_scene_dir)

#set_1 = ['2', '3', '10', '14', '16', '24', '33', '42', '43', '45', '46', '49', '53', '56', '73', '76', '80', '87', '90', '94']
#set_2 = ['5', '22', '50', '29', '31', '36', '68', '69', '70', '78', '86', '97', '99']
#set_3 = ['6', '25' ]
#set_4 = ['7', '13']

image_sets = [['5','22','36'], ['7','13','33'], ['14','24','41'],['99', '25']]  # You can play around with these numbers, refer to the above commented sets and add/remove ref_img ids to add them in scene.

print('Starting fusion....')

for i, test_set in enumerate(image_sets):

    vx = []
    vx_gt = []

    for im in os.listdir(imgs_dir):
        if im[6:] in test_set:
            ply_dir = imgs_dir + '/' + im + '/' + 'ply/'
            ply_gt_dir = imgs_dir + '/' + im + '/' + 'ply_gt/'

            vertices = np.load(ply_dir + 'vertices.npy').reshape(-1,1)
            vx.append(vertices)

            vertices_gt = np.load(ply_gt_dir + 'vertices_gt.npy').reshape(-1,1)
            vx_gt.append(vertices_gt)


    all_vx = np.vstack(tuple(vx)).reshape(-1,)
    all_vx_gt = np.vstack(tuple(vx_gt)).reshape(-1,)

    el = PlyElement.describe(all_vx, 'vertex')
    el_gt = PlyElement.describe(all_vx_gt, 'vertex')

    ply_out_path = fused_scene_dir + '/' + f'scene_{i}.ply'
    ply_out_gt_path = fused_scene_dir + '/' + f'scene_gt_{i}.ply'

    PlyData([el]).write(ply_out_path)
    PlyData([el_gt]).write(ply_out_gt_path)

    print(f'Done fusing for scene {i+1}/{len(image_sets)}...')


