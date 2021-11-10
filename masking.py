'''This file has been used for the research project course (ENGN8601/ENGN8602) by Namas Bhandari of the Australian National University. The file can be found on github.com/namas191297/evaluating_mvs_in_cpc'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import torch.nn.functional as F
import torch
np.printoptions(precision=2)

def mask_loss(gt_mask, pred_mask):

    '''This function is used to calculate the Binary Cross Entropy loss and the IoU between the estimated depthmap
       after mask estimation and the ground truth depthmap'''
  
    gt_mask = torch.from_numpy(gt_mask)
    pred_mask = torch.from_numpy(pred_mask)
    
    loss = F.binary_cross_entropy(pred_mask.float(), gt_mask.float())

    gt_pred = gt_mask > .5
    cv_pred = pred_mask > .5

    intersection = torch.sum(cv_pred & gt_pred, dtype=torch.float32, dim=[0, 1])
    union = torch.sum(cv_pred | gt_pred, dtype=torch.float32, dim=[0, 1])
    gt_sum = torch.sum(gt_pred, dtype=torch.float32, dim=[0, 1])
    cv_sum = torch.sum(cv_pred, dtype=torch.float32, dim=[0, 1])

    acc = torch.mean((cv_pred == gt_pred).to(dtype=torch.float32))
    prec = intersection / cv_sum
    prec[cv_sum == 0] = 1 - intersection.clamp(0, 1)[cv_sum == 0]
    prec = prec.mean()
    rec = intersection / gt_sum
    rec[gt_sum == 0] = 1 - intersection.clamp(0, 1)[gt_sum == 0]
    rec = rec.mean()
    iou = intersection / union
    iou[union == 0] = 1
    iou = iou.mean()

    return {
        "loss": loss,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "iou": iou
    }


def weighted_entropy(entropies, weights):

    '''This function is used to calculate the weighted entropy from the given input entropy maps and the weights'''

    entropies = np.array(entropies)
    entropy = np.zeros((512, 512))
    
    for ix, w in enumerate(weights):
        entropy += np.asarray(entropies[ix]) * w
    
    entropy /= np.sum(weights) 
    return entropy


def plot_grid(ref_img, entropies, op_dir, gt_mask, th_range=10):

    '''This function is used to calculate the 10 best estimated masks for each image and plots it into a grid and saves the mask in a .npy file'''
    
    refcopy = ref_img.copy()
    w_range = np.linspace(0.3, 0.9, 6)
    
    #Load and upsample the entropies
    if entropies == None:
        e_64 = np.load('prob_vol_entropy_64.npy')[0].transpose(1,2,0).reshape(64,64)
        e_64 = scipy.ndimage.zoom(e_64, 8, order=1)
        e_128 = np.load('prob_vol_entropy_128.npy')[0].transpose(1,2,0).reshape(128,128)
        e_128 = scipy.ndimage.zoom(e_128, 4, order=1)
        e_256 = np.load('prob_vol_entropy_256.npy')[0].transpose(1,2,0).reshape(256,256)
        e_256 = scipy.ndimage.zoom(e_256, 2, order=1)
        entropies = [e_64, e_128, e_256]
    else:
        e_64, e_128, e_256 = entropies[0], entropies[1], entropies[2]
    
    # Compute the weighted average of entropies
    #weights = weights
    
    
    # Loop through the weights
    max_loss = 1e+5
    lowest_losses = []
    lowest_ious = []
    mds = []
    
    for w1 in w_range:
        for w2 in w_range:
            for w3 in w_range:
                weights = [w1, w2, w3]
                entropy = weighted_entropy(entropies, weights)
    
                #Save the entropies
                #plt.imsave(f'{op_dir}/{str(w1)+str(w2)+str(w3)}entropy_64.png', e_64)
                #plt.imsave(f'{op_dir}/{str(w1)+str(w2)+str(w3)}entropy_128.png', e_128)
                #plt.imsave(f'{op_dir}/{str(w1)+str(w2)+str(w3)}entropy_256.png', e_256)
    
                #Initialize masks
                mask = np.zeros((512,512))
                threshold_range = np.linspace(1.1, 1.8, 20)
    
    
                #Calculate all masks and plot
                for idx, th in enumerate(threshold_range):
                    mask[np.where(entropy > th)[0], np.where(entropy > th)[1]] = 0
                    mask[np.where(entropy < th)[0], np.where(entropy < th)[1]] = 1


                    #Calculate loss dictionary
                    d = mask_loss(gt_mask, mask)
                    current_loss = d['loss'].numpy().copy()


                    md = {
                        'refcopy':refcopy,
                        'weights':weights,
                        'mask':mask,
                        'entropy':entropy,
                        'th':th,
                        'd':d,
                    }

                    mds.append(md)

                    if current_loss < max_loss:
                        max_loss = current_loss
                        best = {
                            'refcopy':refcopy,
                            'weights':weights,
                            'mask':mask,
                            'entropy':entropy,
                            'th':th,
                            'd':d,
                        }

                        # Calculated occluded pixels
                        valid_pixels = np.where(md['mask'] == 1)
                        missing_pixels = np.where(md['mask'] == 0)

                        # Apply mask
                        masked_image = cv2.bitwise_and(md['refcopy'], md['refcopy'], mask=md['mask'].astype(np.uint8))

                        fig = plt.figure(figsize=(20,10))
                        #plt.title(f'Weights (w1,w2,w3):{weights}, Threshold:{th}')

                        ax1 = fig.add_subplot(1,4,1)
                        ax1.title.set_text('Ref Image')
                        ax1.imshow(md['refcopy'])
                        plt.axis('off')

                        ax2 = fig.add_subplot(1,4,2)
                        ax2.title.set_text('Entropy')
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        ent_im = ax2.imshow(md['entropy'])
                        plt.colorbar(ent_im,fraction=0.04, pad=10)
                        plt.axis('off')

                        ax3 = fig.add_subplot(1,4,3)
                        ax3.title.set_text(f'Mask THR={np.round(md["th"],2)}, Weights={md["weights"]}, Loss:{np.round(md["d"]["loss"],2)}, IoU:{np.round(md["d"]["iou"],2)}')
                        ax3.imshow(md['mask'])
                        plt.axis('off')

                        ax4 = fig.add_subplot(1,4,4)
                        ax4.title.set_text('Masked Image')
                        ax4.imshow(masked_image)
                        plt.axis('off')


                        plt.savefig(op_dir + 'plot' + '_' + str(w1) + str(w2) + str(w3) + '_' + str(md["th"]) + '.png')
                        np.save(f'{op_dir}plot_{str(w1)}{str(w2)}{str(w3)}_{str(md["th"])}.npy', md["mask"])
                        plt.close('all') ##Remove this if any error



    '''The following code plots the best mask obtained as per the losses (However, this can result in worse masks sometimes, so check manually)'''
    #Calculated occluded pixels
    valid_pixels = np.where(best['mask'] == 1)
    missing_pixels = np.where(best['mask'] == 0)

    # Apply mask
    masked_image = cv2.bitwise_and(best['refcopy'], best['refcopy'], mask=best['mask'].astype(np.uint8))

    fig = plt.figure(figsize=(20,10))
    #plt.title(f'Weights (w1,w2,w3):{weights}, Threshold:{th}')
    
    # Append the best loss
    lowest_losses.append(np.round(best["d"]["loss"],2))
    lowest_ious.append(np.round(best["d"]["iou"],2))
    

    # Plots
    ax1 = fig.add_subplot(1,4,1)
    ax1.title.set_text('Ref Image')
    ax1.imshow(best['refcopy'])
    plt.axis('off')

    ax2 = fig.add_subplot(1,4,2)
    ax2.title.set_text('Entropy')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ent_im = ax2.imshow(best['entropy'])
    plt.colorbar(ent_im,fraction=0.04, pad=10)
    plt.axis('off')

    ax3 = fig.add_subplot(1,4,3)
    ax3.title.set_text(f'Mask THR={np.round(best["th"],2)}, Weights={best["weights"]}, Loss:{np.round(best["d"]["loss"],2)}, IoU:{np.round(best["d"]["iou"],2)}')
    ax3.imshow(best['mask'])
    plt.axis('off')

    ax4 = fig.add_subplot(1,4,4)
    ax4.title.set_text('Masked Image')
    ax4.imshow(masked_image)
    plt.axis('off')


    plt.savefig(op_dir + 'best' + '.png')
    plt.close('all') ##Remove this if any error
    
    return np.round(best["d"]["loss"],2), np.round(best["d"]["iou"],2)

if __name__ == '__main__':
    
    print('Starting Mask Estimation.....')

    r_dir = 'ref_images/'
    lowest_losses = []
    lowest_ious = []

    for idx, imf in enumerate(os.listdir(r_dir)):
    
        imf_dir = r_dir + imf + '/'
        ref_image = None

        if os.path.exists(imf_dir + 'grid_outputs'):
            for f in os.listdir(imf_dir + 'grid_outputs'):
                os.remove(imf_dir + 'grid_outputs' + '/' + f)

            os.rmdir(imf_dir + 'grid_outputs')
            os.mkdir(imf_dir + 'grid_outputs')

        else:
            op_dir = os.mkdir(imf_dir + 'grid_outputs')
  
        grid_dir = imf_dir + 'grid_outputs/'
    
        for item in os.listdir(imf_dir):
            item_path = imf_dir + item
            #gt_path = imf_dir + 'label.png' 
            gt_path = imf_dir + 'mask.npy'
        
            if 'refimage' in item and 'png' in item:
                ref_image = plt.imread(item_path)[:,:,:3]
            elif '64' in item:
                e_64 = np.load(item_path)[0].transpose(1,2,0).reshape(64,64)
                e_64 = scipy.ndimage.zoom(e_64, 8, order=1)
            elif '128' in item:
                e_128 = np.load(item_path)[0].transpose(1,2,0).reshape(128,128)
                e_128 = scipy.ndimage.zoom(e_128, 4, order=1)
            elif '256' in item:
                e_256 = np.load(item_path)[0].transpose(1,2,0).reshape(256,256)
                e_256 = scipy.ndimage.zoom(e_256, 2, order=1)
    

        #gt_mask = cv2.imread(gt_path)[:,:,2]    
        #gt_mask = np.where(gt_mask == 0, 1, 0)  
    
        gt_mask = np.load(gt_path)[0][0]
    
        #plt.imshow(gt_mask)
        #plt.colorbar()
        #plt.show()
        entropies = [e_64, e_128, e_256]
            
        lowest_loss, lowest_iou = plot_grid(ref_image, entropies, grid_dir, gt_mask)
        lowest_losses.append(lowest_loss)
        lowest_ious.append(lowest_ious)
    
        print(f'Done plotting for Image {idx+1}/{len(os.listdir(r_dir))}: Loss:{lowest_loss}, IoU:{lowest_iou}')

    print(f'Average Loss:{np.array(lowest_losses).mean()}, Average IoU:{np.array(lowest_ious).mean()}')

