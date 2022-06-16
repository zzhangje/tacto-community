import torch
import torch
import torch.nn.functional
from .fcrn_helpers.fcrn import FCRN_net
from .fcrn_helpers.loader import TestDataLoader
import numpy as np
import cv2

from .TactoRender import TactoRender, pixmm
from .utils.util3D import Util3D
from shapeclosure.misc import *
from PIL import Image
import pickle 
import collections 
from shapeclosure.misc import plotSubplots

class fcrn:
    def __init__(self, weights_path, bg, blend_sz = 0, bottleneck = False, real = False, gpu = True):

        # print("setting devices...")
        use_cuda = torch.cuda.is_available() if gpu else False
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        if real: 
            self.b, self.r, self.clip = 10, 0.8, 5
        else:
            self.b, self.r, self.clip = 1, 0.2, 5        
        # print("setting parameters...")
        self.batch_size = 1
        self.params = {'batch_size': self.batch_size, 'shuffle': False}

        self.model = FCRN_net(self.batch_size, bottleneck = bottleneck)
        checkpoint = torch.load(weights_path, map_location=self.device)
        print("=> loaded fcrn (epoch {})".format(checkpoint['epoch']))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)

        self.blend_sz = blend_sz
        self.heightmap_window = collections.deque([])

        self.bg = bg

    def blend_heightmaps(self, heightmap):
        if not self.blend_sz: 
            return heightmap
        
        if len(self.heightmap_window) >= self.blend_sz:
            self.heightmap_window.popleft()
        
        self.heightmap_window.append(heightmap)
        n = len(self.heightmap_window)

        # exponentially weighted time series costs 
        weights = np.array([x/n for x in range(1, n+1)])
        weights = np.exp(weights) / np.sum(np.exp(weights))

        blended_heightmap = np.average(list(self.heightmap_window), axis=0, weights=weights)
        # plotSubplots([heightmap, blended_heightmap], [["heightmap", "blended_heightmap"]])
        return blended_heightmap

    def image2heightmap(self, test_data):
        # test_data: tactile img 640 * 480
        # result: height map 640 * 480
        # test_data = cv2.cvtColor(test_data, cv2.COLOR_RGB2BGR) 
        assert self.model.bottleneck is False, "Bottleneck feature is enabled, can't carry out image2heightmap"

        test_data = cv2.normalize(test_data, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
        # test_data = cv2.GaussianBlur(test_data,(15,15),cv2.BORDER_DEFAULT)

        test_set = TestDataLoader(test_data)
        test_loader = torch.utils.data.DataLoader(test_set, **self.params)
        with torch.no_grad():
            for data in test_loader:
                data = data.type(torch.FloatTensor).to(self.device)
                output = self.model(data)[0].data.cpu().squeeze().numpy().astype(np.float32)
                if (output.shape[0] == 240):
                    output = cv2.resize(output, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
                return self.blend_heightmaps(output)

    def image2embedding(self, test_data):
        if self.model.bottleneck is False:
            print("Bottleneck feature extraction not enabled, switching")
            self.model.bottleneck = True
        # test_data: tactile img 640 * 480
        test_data = cv2.normalize(test_data, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
        # test_data = cv2.GaussianBlur(test_data,(15,15),cv2.BORDER_DEFAULT)
        test_set = TestDataLoader(test_data)
        test_loader = torch.utils.data.DataLoader(test_set, **self.params)
        with torch.no_grad():
            for data in test_loader:
                data = data.type(torch.FloatTensor).to(self.device)
                output = self.model(data)[0].data.cpu().squeeze().numpy().astype(np.float16)
                feature = output.reshape((-1,10*8*1024))
                return feature

    def heightmap2mask(self, heightmap):
        heightmap = heightmap[self.b:-self.b,self.b:-self.b]
        init_height = self.bg[self.b:-self.b,self.b:-self.b]
        diff_heights = heightmap - init_height
        diff_heights[diff_heights<self.clip]=0
        contact_mask = diff_heights > np.percentile(diff_heights, 90) * self.r
        padded_contact_mask = np.zeros(self.bg.shape, dtype=bool)

        if np.count_nonzero(contact_mask) < 0.1*(contact_mask.shape[0] * contact_mask.shape[1]):
            return padded_contact_mask
        padded_contact_mask[self.b:-self.b,self.b:-self.b] = contact_mask
        return padded_contact_mask

if __name__ == "__main__":
    obj_name, log_id = '035_power_drill', 0
    data_path = osp.join("/home/suddhu/projects/fair-3d/shape-closures/data", obj_name, str(log_id).zfill(2))
    obj_path = osp.join("/home/suddhu/projects/fair-3d/shape-closures/models", obj_name, "google_512k/nontextured.stl")

    image_path, pose_path = osp.join(data_path, "tactile_images"), osp.join(data_path, "tactile_data.pkl") 
    heightmap_path, contactmask_path = osp.join(data_path, "gt_heightmaps"), osp.join(data_path, "gt_contactmasks")

    u3d = Util3D(obj_path, off_screen = False, virtual_buff = False)

    FCRNModel = fcrn()
    tacRender = TactoRender(obj_path=obj_path, headless = True)

    # load images and ground truth depthmaps  
    imageFiles = sorted(os.listdir(image_path), key=lambda y: int(y.split(".")[0]))
    heightmapFiles = sorted(os.listdir(heightmap_path), key=lambda y: int(y.split(".")[0]))
    contactmaskFiles = sorted(os.listdir(contactmask_path), key=lambda y: int(y.split(".")[0]))

    # poses
    if osp.exists(pose_path):
        print("Loading poses: {}".format(pose_path))
        with open(pose_path, 'rb') as pickle_file:
            poseDict = pickle.load(pickle_file)
        camposes, gelposes = poseDict["camposes"], poseDict["gelposes"]
    else:
        print("Pose path not available")

    N = len(imageFiles)
    for i in range(N): 
        # Open images
        image = np.array(Image.open(osp.join(image_path, imageFiles[i])))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        gt_heightmap = np.array(Image.open(osp.join(heightmap_path, heightmapFiles[i]))).astype(np.int64)
        contactmask = np.array(Image.open(osp.join(contactmask_path, contactmaskFiles[i]))).astype(bool)

        # Convert image to heightmap via lookup
        est_heightmap = FCRNModel.image2heightmap(image)
        est_contactmask = FCRNModel.heightmap2mask(est_heightmap)
        # Get pixelwise RMSE in mm, and IoU of the contact masks
        error_heightmap = np.abs(est_heightmap - gt_heightmap) * pixmm
        heightmap_rmse  = np.sqrt(np.mean(error_heightmap**2))
        intersection = np.sum(np.logical_and(contactmask, est_contactmask))
        contact_mask_iou = intersection/(np.sum(contactmask) + np.sum(est_contactmask) - intersection)
        
        # Visualize heightmaps
        print("Heightmap RMSE: {:.4f} mm, Contact mask IoU: {:.4f}".format(heightmap_rmse, contact_mask_iou))
        plotSubplots( [image/255.0, gt_heightmap, contactmask, est_heightmap, est_contactmask, error_heightmap], 
                      [['Tactile image', 'GT heightmap', 'GT contact mask'], ['Est. heightmap', 'Est. contact mask', 'Heightmap Error (mm']])

        # Convert heightmaps to 3D
        gt_cloud = tacRender.heightmap2Pointcloud(gt_heightmap, contactmask)
        gt_cloud_w = transformPointcloud(gt_cloud.copy(), camposes[i])
        est_cloud = tacRender.heightmap2Pointcloud(est_heightmap, est_contactmask)
        est_cloud_w = transformPointcloud(est_cloud.copy(), camposes[i])
        # Visualize results
        u3d.vizMeasurements(np.vstack((camposes[i], gelposes[i])), [est_cloud_w] , annotations = None, save_path = None, decimation_factor = 1)