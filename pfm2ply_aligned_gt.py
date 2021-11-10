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

def unproject_depth(depth, intrinsics, extrinsics):
    '''
        Project a given depth map into 3D based on given camera intrinsic and extrinsic parameters.
        And assign vertex color by color from image.
        Input:
            depth: a HxW array.
            intrinsics: a 3x3 array, the K matrix.
            extrinsics: a 3x4 array, the RT matrix.
        Output:
            xyz_world: a [3,N] array of N pts in world 3D coordinate.
    '''

    width, height = depth.shape[1], depth.shape[0]

    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

    # image space -> reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth.reshape([-1]))

    # reference 3D space -> world 3D space
    xyz_world = np.matmul(np.linalg.inv(extrinsics),np.vstack((xyz_ref,np.ones_like(x_ref))))

    # Normalize
    xyz_world = xyz_world/xyz_world[3]

    return xyz_world[:3]

main_dir = 'ref_images/'
ply_dir = 'ply_gt'    # For ground truth output folder

for idx, im in enumerate(os.listdir(main_dir)): 

    print(f'---------- Image {idx+1}/{len(os.listdir(main_dir))} ----------')

    curr_dir = main_dir + im + "/"
    
    if os.path.exists(curr_dir + ply_dir):
        for f in os.listdir(curr_dir + ply_dir):
            os.remove(curr_dir + ply_dir + '/' + f)
        os.rmdir(curr_dir + ply_dir)
        os.mkdir(curr_dir + ply_dir)
    else:
        os.mkdir(curr_dir + ply_dir)


    est_dp = np.load(curr_dir+'est_depthmap.npy')
    gt_dp = np.load(curr_dir+'gt_depthmap.npy')[0][0]
    mono_dp = np.load(curr_dir+'mono.npy')
    mask = np.load(curr_dir+'est_mask.npy')[0][0]
    gt_mask = np.load(curr_dir+'mask.npy')[0][0]

    # Upsample the estimated map to gt size
    est_dp = torch.from_numpy(est_dp)
    up_est_dp = F.interpolate(est_dp, scale_factor=2, mode="bilinear").numpy()[0][0]

    # Cam
    K = np.load(curr_dir+'K.npy')
    R = np.load(curr_dir+'R.npy')
    t = np.load(curr_dir+'t.npy')
    depth_min, depth_max = np.load(curr_dir+'depth_min.npy')[0][0], np.load(curr_dir+'depth_max.npy')[0][0]
    depth_interval = (depth_max - depth_min) / 128
    depth_max = depth_min + (256 * depth_interval)

    # Create the extrinsic matrix
    R_t = np.zeros((4,4))
    R_t[:3,:3] = R
    R_t[:3,3] = t.T
    R_t[3, :] = [0, 0, 0, 1]

    img = Image.open(f'{curr_dir}refimage_{im[6:]}.png')
    inverse_est_depth = mono_dp.copy()
    ref_depth = up_est_dp.copy()
    intrinsics, extrinsics = K, R_t

    width, height = inverse_est_depth.shape[1], inverse_est_depth.shape[0]
    ref_width, ref_height = ref_depth.shape[1], ref_depth.shape[0]

    down_ratio = 1
    ply_out_path = curr_dir + ply_dir + '/' + 'output_gt.ply'      # For ground truth output

    xyz_world = unproject_depth(ref_depth, intrinsics, extrinsics)     # For the ground truth depthmap
    
    # Create depth vector to filter for ground truth 
    depth_vec = gt_mask.reshape(1,-1)   

    xyz_world_down = xyz_world[:,::down_ratio]

    # Make point cloud
    # Vertices
    vertexs = xyz_world_down.transpose((1, 0))
    vertexs = np.array([tuple(v) for i,v in enumerate(vertexs) if depth_vec[0,i] != 0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Image color
    # make color different from gt

    #img = img.astype(np.float32)
    #img[:,:,1:3] *= 0.7
    #img = img.astype(np.uint8)


    color = np.array(img)[:,:,:3].reshape([width*height,3])[depth_vec[0,:] != 0, :]
    color_down = color[::down_ratio,:]
    vertex_colors = color_down.astype(np.uint8)
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    print(f'Saving Image {idx+1} ground truth vertices....')
    np.save(f'{curr_dir + ply_dir + "/"}vertices_gt.npy', vertex_all)

    el = PlyElement.describe(vertex_all, 'vertex')
    print(f"Saving Image {idx+1} ground truth point cloud...")
    PlyData([el]).write(ply_out_path)

    print(f'---------- Image {idx+1}/{len(os.listdir(main_dir))} DONE! ----------')

    
