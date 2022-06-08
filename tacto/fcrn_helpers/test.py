import torch
import torch.utils.data
from loader import RealDataLoader, GelDataLoader
import numpy as np
import os
from os import path as osp
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from tqdm import tqdm 
from tacto.TactoRender import TactoRender, pixmm
from shapeclosure.misc import *
from tacto.fcrn import fcrn


dtype = torch.cuda.FloatTensor

def test_real():
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    checkpoint_path = './weights/checkpoint_heightmap_digit.pth.tar'
    data_file_path = osp.join("data_files")
    test_results_path = "/mnt/sda/suddhu/fcrn/fcrn-testing"

    test_real_file = osp.join(data_file_path,'test_data_real.txt')
    test_loader = torch.utils.data.DataLoader(RealDataLoader(test_real_file), batch_size=50, shuffle=False, drop_last=True)

    device = getDevice(cpu = False)

    tacRender = TactoRender(obj_path = None,  headless = True)
    # Load FCRN weights
    print(f'Loading weights: {checkpoint_path}')
    bg = tacRender.get_background(frame = 'gel')
    FCRNModel = fcrn(checkpoint_path, bg)
    FCRNModel.to(device)

    if not osp.exists(test_results_path):
        os.makedirs(test_results_path)

    # test on real dataset 
    print('Testing on real data')
    with torch.no_grad():
        count = 0
        pbar = tqdm(total = len(test_loader))
        for input in test_loader:
            input_var = Variable(input.type(dtype))
            for i in range(len(input_var)):
                input_rgb_image = input_var[i].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                est_h = FCRNModel.image2heightmap(input_rgb_image)
                est_c  = FCRNModel.heightmap2mask(est_h)
                # pred_image /= np.max(pred_image)
                plot.imsave(osp.join(test_results_path, "{count}_input.png"), input_rgb_image)
                plot.imsave(osp.join(test_results_path, "{count}_pred_heightmap.png"), est_h, cmap="viridis")
                plot.imsave(osp.join(test_results_path, "{count}_pred_mask.png"), est_c)
                count += 1
            pbar.update(1)
        pbar.close()
    return 

def test_sim():
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    checkpoint_path = './weights/checkpoint_heightmap_digit_sim.pth.tar'

    data_file_path = osp.join("data_files")
    test_results_path = "/mnt/sda/suddhu/fcrn/fcrn-testing"
    test_data_file = osp.join(data_file_path,'test_data.txt')
    test_label_file = osp.join(data_file_path,'test_label.txt')

    test_loader = torch.utils.data.DataLoader(GelDataLoader(test_data_file, test_label_file),
                                               batch_size=50, shuffle=False, drop_last=True)
    device = getDevice(cpu = False)

    tacRender = TactoRender(obj_path = None,  headless = True)
    # Load FCRN weights
    print(f'Loading weights: {checkpoint_path}')
    bg = tacRender.get_background(frame = 'gel')
    FCRNModel = fcrn(checkpoint_path, bg)
    FCRNModel.to(device)

    if not osp.exists(test_results_path):
        os.makedirs(test_results_path)

    heightmap_rmse, contact_mask_iou = [], []

    # test on real dataset 
    print('Testing on sim data')
    with torch.no_grad():
        count = 0
        pbar = tqdm(total = len(test_loader))
        for input, depth in test_loader:
            input_var = Variable(input.type(dtype))
            gt_var = Variable(depth.type(dtype))

            for i in range(len(input_var)):
                input_rgb_image = input_var[i].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                gt_c = gt_var[i].data.squeeze().cpu().numpy().astype(np.float32)

                est_h = FCRNModel.image2heightmap(input_rgb_image)
                est_c  = FCRNModel.heightmap2mask(est_h)

                error_heightmap = np.abs(est_h - gt_c) * pixmm
                heightmap_rmse.append(np.sqrt(np.mean(error_heightmap**2)))
                intersection = np.sum(np.logical_and(gt_c, est_c))
                contact_mask_iou.append(intersection/(np.sum(est_c) + np.sum(gt_c) - intersection))
                count += 1
            pbar.update(1)
        pbar.close()

        heightmap_rmse = [x for x in heightmap_rmse if str(x) != 'nan']
        contact_mask_iou = [x for x in contact_mask_iou if str(x) != 'nan']
        heightmap_rmse = sum(heightmap_rmse) / len(heightmap_rmse)
        contact_mask_iou = sum(contact_mask_iou) / len(contact_mask_iou)
        error_file = open(osp.join(test_results_path,'fcrn_error.txt'),'w')
        error_file.write(str(heightmap_rmse) + "," + str(contact_mask_iou) + "\n")
        error_file.close()

    return 

if __name__ == '__main__':
    # test_real()
    test_sim()
