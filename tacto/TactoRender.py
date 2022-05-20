'''
TACTO rendering class
'''

import os
from os import path as osp
from tkinter import W
import numpy as np
from .utils.utils import *
from .loaders.ObjectLoader import ObjectLoader
import tacto 
from .renderer import euler2matrix, DEBUG
from .utils.utils import *
import cv2 

import trimesh 

pixmm = 0.03 # 0.0295; # 0.0302

class TactoRender:
    def __init__(self, obj_path = None, randomize : bool = False, headless = False):
        # Create renderer
        self.renderer = tacto.Renderer(width=240, height=320, background= cv2.imread(tacto.get_background_image_path(0)), config_path=tacto.get_digit_shadow_config_path(), headless = headless)

        self.cam_dist = 0.022

        if not DEBUG:
            _, self.bg_depth = self.renderer.render()
            self.bg_depth = self.bg_depth[0]
            self.bg_depth_pix = self.correct_pyrender_height_map(self.bg_depth)

        if obj_path is not None:
            self.obj_loader = ObjectLoader(obj_path)
            obj_trimesh = trimesh.load(obj_path)
            self.renderer.add_object(obj_trimesh, "object")
        
        self.press_depth = 0.001
        self.randomize = randomize
    
    def get_background(self, frame = 'gel'):
        if frame == 'gel':
            return self.bg_depth_pix
        else:
            self.bg_depth
    
    def update_pose_given_pose(self, press_depth, gelpose):
        self.press_depth = press_depth
        campose = self.gel2cam(gelpose)
        campose = self.addPenetration(campose)
        self.renderer.update_camera_pose_from_matrix(self.switchAxes(campose))
        # self.renderer.update_camera_pose_from_matrix(campose)

    def pix2meter(self, pix):
        return pix * pixmm / 1000.0
    
    def meter2pix(self, m):
        return m * 1000.0 / pixmm
        
    def update_pose_given_point(self, point, press_depth, shear_mag, delta):
        dist = np.linalg.norm(point - self.obj_loader.obj_vertices, axis=1)
        idx = np.argmin(dist)

        # idx: the idx vertice, get a new pose
        new_position = self.obj_loader.obj_vertices[idx].copy()
        new_orientation = self.obj_loader.obj_normals[idx].copy()

        delta = np.random.uniform(low=0.0, high=0.08, size=(1,))[0]
        new_pose = gen_pose(new_position, new_orientation, shear_mag, delta).squeeze()
        self.update_pose_given_pose(press_depth, new_pose)

    def switchAxes(self, pose):
        # Opposite of transformation in config_digit_shadow.yml
        switch_axes = euler2matrix(angles = [-90, 0, 90], xyz="zyx", degrees=True)
        return np.matmul(pose, switch_axes)
    
    def addPenetration(self, pose):
        pen_mat = np.eye(4)
        pen_mat[2,3] = -self.press_depth
        return np.matmul(pose, pen_mat) 

    def gel2cam(self, gel_pose):
        cam_tf = np.eye(4)
        cam_tf[2, 3] = self.cam_dist
        return np.matmul(gel_pose, cam_tf) 

    def cam2gel(self, cam_pose):
        cam_tf = np.eye(4)
        cam_tf[2, 3] = -self.cam_dist
        return np.matmul(cam_pose, cam_tf) 

    # input depth is in camera frame here 
    def render(self):
        color, depth = self.renderer.render()
        color, depth = color[0], depth[0]
        diff_depth = ((self.bg_depth) - depth)
        contact_mask = diff_depth > np.abs(self.press_depth * 0.2)
        gel_depth = self.correct_pyrender_height_map(depth) #  pix in gel frame
        # cam_depth = self.correct_image_height_map(gel_depth) #  pix in gel frame
        # assert np.allclose(cam_depth, depth), "Conversion to pixels is incorrect"
        if self.randomize:
            self.renderer.randomize_light()
        return color, gel_depth, contact_mask

    def correct_pyrender_height_map(self, height_map):
        '''
        Input: height_map in meters, in camera frame
        Output: height_map in pixels, in gel frame
        '''
        # move to the gel center
        height_map = (self.cam_dist - height_map) * (1000 / pixmm)
        return height_map

    def correct_image_height_map(self, height_map, output_frame = 'cam'):
        '''
        Input: height_map in pixels, in gel frame
        Output: height_map in meters, in camera/gel frame
        '''
        height_map = -height_map * (pixmm / 1000) + float(output_frame == 'cam') * self.cam_dist
        return height_map

    def get_cam_pose_matrix(self):
        return self.renderer.camera_nodes[0].matrix

    def get_cam_pose(self):
        # print(f"Cam pose: {gen_t_quat(self.get_cam_pose_matrix())}")
        return gen_t_quat(self.get_cam_pose_matrix())

    def get_gel_pose_matrix(self):
        return self.cam2gel(self.get_cam_pose_matrix())

    def get_gel_pose(self):
        # print(f"Gel pose: {gen_t_quat(self.get_gel_pose_matrix())}")
        return gen_t_quat(self.get_gel_pose_matrix())

    def heightmap2Pointcloud(self, depth, contact_mask = None):
        ''' Convert heightmap + contact mask to point cloud
        Heightmap can be straight from pyrender, or generated via lookup table
        [Input]  depth: 640 x 480 in pixels, in gel frame, Contact mask: binary 640 x 480
        [Output] pointcloud: [(640x480) - (masked off points), 3] in meters in camera frame
        '''
        depth = self.correct_image_height_map(depth, output_frame='cam')

        if contact_mask is not None:
            heightmapValid = np.multiply(depth, contact_mask) # apply contact mask
        else:
            heightmapValid = depth

        f, w, h = self.renderer.f, self.renderer.width/2., self.renderer.height/2.

        # (0, 640) and (0, 480)
        xvals = np.arange(heightmapValid.shape[1])
        yvals = np.arange(heightmapValid.shape[0])
        [x,y] = np.meshgrid(xvals, yvals)

        # x and y in meters
        x = ((x  - w))/f
        y = ((y - h))/f

        x *= depth
        y *= -depth

        heightmap_3d = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)), heightmapValid.reshape((-1, 1))))
        
        heightmap_3d[:, 2] *= -1
        heightmap_3d = heightmap_3d[heightmap_3d[:, 2] != 0]

        # import matplotlib.pyplot as plt
        # from mpl_toolkits import mplot3d
        # fig = plt.figure()
        # ax = plt.subplot(1, 1, 1, projection='3d')
        # # ax.set_title("Point cloud", size=10)
        # ax.scatter3D(heightmap_3d[::1, 0], heightmap_3d[::1, 1], heightmap_3d[::1, 2], c=heightmap_3d[::1, 2], cmap='viridis', s = .1)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # fig.tight_layout()

        # plt.show(block=True)
        # plt.close()

        return heightmap_3d
        
if __name__ == "__main__":
    import configparser

    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    config_file = "../../shapeclosure/config/config.ini"

    obj_model = "035_power_drill"
    config = configparser.ConfigParser()
    config.read(config_file)
    use_cluster = eval(config['DEFAULT'].get('use_cluster'))
    params = config['CLUSTER'] if use_cluster else config['CPU']

    obj_model_path = osp.join(params.get('model_path'), obj_model)
    obj_path = osp.join(obj_model_path, "google_512k", "nontextured.stl")

    press_depth = 0.001 # in meter
    vertix_idxs = np.random.choice(1000, size = 500) # [159] 
    # try both sensor models over different vertices 
    tacRender = TactoRender(obj_path, randomize = True, headless = use_cluster)

    from PIL import Image
    images = []
    vertix_idxs = [857] * 100
    for vertix_idx in vertix_idxs:
        tacRender.update_pose_given_vertex(vertix_idx, press_depth, shear_mag = 0.0)
        tactile_img, height_map, contact_mask  = tacRender.render()
        # plotSubplots([0.022 - height_map, tactile_img/255.0, contact_mask], 
        #              [["v : {} Heightmap DIGIT".format(vertix_idx), "Tactile image DIGIT", "Contact Mask DIGIT"]])
        images.append(Image.fromarray(tactile_img))
    images[0].save("augmentations.gif", save_all=True, append_images=images, duration=200, loop=0)
