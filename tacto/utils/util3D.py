'''
utility functions

Sudharshan Suresh (suddhu@cmu.edu)
Last revision: Oct 2021
'''

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import scipy.spatial
import open3d as o3d 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import cv2
import os 
from PIL import Image
from .utils import gen_quat_t
from tqdm import tqdm 
from pyvistaqt import BackgroundPlotter
import trimesh 
import copy 
import open3d as o3d

pv.set_plot_theme("document")
# pv.global_theme.axes.show = True
# pv.global_theme.axes.x_color = 'black'

class Util3D:
    def __init__(self, mesh_path, off_screen = True, virtual_buff = False):
        if virtual_buff:
            pv.start_xvfb()  

        self.mesh_path = mesh_path
        mesh = trimesh.load(mesh_path)
        mesh = mesh.simplify_quadratic_decimation(face_count = int(mesh.vertices.shape[0]/10)) #pv.read(self.mesh_path)
        self.deci_mesh = pv.wrap(mesh)
        self.mesh = pv.read(mesh_path)

        self.framerate = 10        

        # load and rotate gelsight mesh 
        if virtual_buff:
            self.gelsight_mesh = pv.read("/home/rpluser/Documents/suddhu/projects/shape-closures/models/digit/digit.STL")
        else:
            self.gelsight_mesh = pv.read("/home/suddhu/rpl/datasets/YCBModels/digit/digit.STL")
        # T = np.eye(4)
        # T[:3,:3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        # self.gelsight_mesh.rotate_y(90, point=self.gelsight_mesh.center, inplace = True)

        self.virtual_buff = virtual_buff
        # self.gelsight_mesh = self.gelsight_mesh.transform(T, inplace = False)
        self.samplesActor, self.gelsightActorA, self.gelsightActorB, self.heightmapActor, self.imageActor, self.heatActor = None, None, None, None, None, None
        self.meanQuiverXActor, self.meanQuiverYActor, self.meanQuiverZActor, self.ellipsoidActor, self.frame_text = None, None, None, None, None

        self.off_screen = off_screen
        pv.global_theme.multi_rendering_splitting_position = 0.7

        shape = (2, 2)  # 5 by 4 grid
        # First row is half the size and fourth row is double the size of the other rows
        row_weights = [0.5, 0.5]
        # Third column is half the size and fourth column is double size of the other columns
        col_weights = [0.6, 0.4]
        groups = [
            (np.s_[:], 0),  # First group spans over all columns of the first row (0)
            (0, 1),  # Second group spans over row 1-3 of the first column (0)
            (1, 1),  # Second group spans over row 1-3 of the first column (0)
        ]
        if not off_screen:
            # self.p = BackgroundPlotter(shape='1|2', border_color = "white", border_width = 0, off_screen=self.off_screen, window_size=(1920, 1088))
            self.p = BackgroundPlotter(lighting='three lights', shape=shape, row_weights=row_weights, col_weights=col_weights, groups=groups, window_size=(1760, 1200), border_color = "white")
        else:
            self.p = pv.Plotter(lighting='three lights', shape=shape, off_screen=self.off_screen, row_weights=row_weights, col_weights=col_weights, groups=groups, window_size=(1760, 1200), border_color = "white")
        # print(self.p.ren_win.ReportCapabilities())
    
    def exit(self):
        self.p.close()
    
    def screenshotModel(self, savepath):
        p = pv.Plotter(off_screen=self.off_screen, window_size=(1200, 1200))
        dargs = dict(smooth_shading=True, specular=1.0, show_scalar_bar=False, silhouette=True)
        p.add_mesh(self.mesh, **dargs)
        p.set_focus(self.mesh.center)
        p.camera_position, p.camera.azimuth, p.camera.elevation = 'yz', 45, 20
        p.camera.Zoom(1)
        p.screenshot(savepath, transparent_background = True)  
        p.close()

    def initDensityMesh(self, gt_pose, save_path):

        if self.virtual_buff: 
            return
        self.p.subplot(0, 0)
        gt_pose = np.atleast_2d(gt_pose)
        dargs = dict(color="grey", ambient=0.6, opacity=0.5, smooth_shading=True, specular=1.0, show_scalar_bar=False)
        self.p.add_mesh(self.mesh, **dargs)
        # self.p.set_focus(self.mesh.center)
        self.p.camera_position, self.p.camera.azimuth, self.p.camera.elevation = 'yz', 45, 20
        self.p.camera.Zoom(1)
        self.p.camera_set = True

        traj_plot = gt_pose[:, :3].copy()
        (traj_plot, _, _) = trimesh.proximity.closest_point(trimesh.load(self.mesh_path), traj_plot)
        spline = pv.Spline(traj_plot, 500)
        self.p.add_mesh(spline, line_width=3, color="k")        # plot without scalars

        self.p.subplot(0, 1)
        self.p.camera.Zoom(1)

        if not self.off_screen:
            self.p.show() 
        self.p.open_movie(save_path + ".mp4", framerate=self.framerate)
        print(f"Animating particle filter at {save_path}.mp4")

    def updateHeatmap(self, samples, heat, heat_weights):
        if self.virtual_buff: 
            return

        self.p.subplot(0, 0)
        samplePoints = pv.PolyData(samples[:, :3])     
        dargs = dict(color = 'red',  show_scalar_bar=False, opacity=0.3, reset_camera = False)
        self.samplesActor = self.p.add_mesh(samplePoints, render_points_as_spheres=True, point_size=5, **dargs)

        self.p.subplot(1, 1)
        self.p.remove_actor(self.heatActor)
        heatCloud = pv.PolyData(heat[:, :3])
        heat_weights = (heat_weights - np.min(heat_weights)) / (np.max(heat_weights) - np.min(heat_weights))        
        heat_weights[heat_weights < np.percentile(heat_weights, 80)] = 0

        heatCloud["similarity"] = heat_weights
        HeatmapMesh = self.deci_mesh.interpolate(heatCloud, strategy="null_value", radius = self.mesh.length/50)
        # viridis.colors[0] =  [189/256, 189/256, 189/256] # grey
        dargs = dict(cmap = cm.get_cmap('viridis'), scalars='similarity', interpolate_before_map = True, ambient = 0.2, opacity=0.9,  show_scalar_bar=False, silhouette=True)
        self.heatActor = self.p.add_mesh(HeatmapMesh, **dargs)

        self.p.set_focus(self.deci_mesh.center)
        self.p.camera_position, self.p.camera.azimuth, self.p.camera.elevation = 'yz', 45, 20
        self.p.camera.Zoom(0.9)
        self.p.camera_set = True

        return

    def updateDensityMesh(self, _particles, cluster_poses, cluster_stds, gt_pose, heat, heat_weights, image, heightmap, mask, count, image_savepath = None):
        if self.virtual_buff: 
            return
        
        particles = copy.copy(_particles)

        samples = np.atleast_2d(particles.poses)

        if self.samplesActor:
            self.p.remove_actor(self.samplesActor)
        if self.imageActor:
            self.p.remove_actor(self.imageActor)
        if self.heightmapActor:
            self.p.remove_actor(self.heightmapActor)
        if self.gelsightActorA:
            self.p.remove_actor(self.gelsightActorA)
        if self.gelsightActorB:
            self.p.remove_actor(self.gelsightActorB)
        if self.heatActor:
            self.p.remove_actor(self.samplesActor)

        if self.meanQuiverXActor:
            self.p.remove_actor(self.meanQuiverXActor)
        if self.meanQuiverYActor:
            self.p.remove_actor(self.meanQuiverYActor)
        if self.meanQuiverZActor:
            self.p.remove_actor(self.meanQuiverZActor)

        if self.frame_text:
            self.p.remove_actor(self.frame_text)

        if self.ellipsoidActor:
            for e in self.ellipsoidActor: 
                self.p.remove_actor(e)

        self.p.subplot(0, 0)
           
        # visualize gelsight 
        transformed_gelsight_mesh = self.gelsight_mesh.transform(gen_quat_t(gt_pose), inplace = False)
        dargs = dict(color = 'tan', ambient=0.0, opacity=0.7, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
        self.gelsightActorA = self.p.add_mesh(transformed_gelsight_mesh, **dargs)

        sigma_3 = 3. * np.mean(cluster_stds, axis = 1)
        invalid = sigma_3 > self.mesh.length / 10
        cluster_poses = np.delete(cluster_poses, invalid, axis = 0)
        cluster_stds = np.delete(cluster_stds, invalid, axis = 0)

        n_clusters = cluster_poses.shape[0]
        if 0 < n_clusters < 10: 
            self.ellipsoidActor = [None] * n_clusters
            for i, (cluster_pose, cluster_std) in enumerate(zip(cluster_poses, 3 * cluster_stds)):
                var_ellipsoid = pv.ParametricEllipsoid(cluster_std[0], cluster_std[1], cluster_std[2])
                var_ellipsoid.translate(cluster_pose[:3], inplace=True)
                dargs = dict(color = "red", ambient=0.0, opacity=0.2, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
                self.ellipsoidActor[i] = self.p.add_mesh(var_ellipsoid, **dargs)
    
            cluster_quivers = pose2quiver(cluster_poses, self.mesh.length / 10)
            cluster_quivers.set_active_vectors("xvectors")
            self.meanQuiverXActor = self.p.add_mesh(cluster_quivers.arrows, color="red", show_scalar_bar=False, opacity=0.5)
            cluster_quivers.set_active_vectors("yvectors")
            self.meanQuiverYActor = self.p.add_mesh(cluster_quivers.arrows, color="green", show_scalar_bar=False, opacity=0.5)
            cluster_quivers.set_active_vectors("zvectors")
            self.meanQuiverZActor = self.p.add_mesh(cluster_quivers.arrows, color="blue", show_scalar_bar=False, opacity=0.5)
            
        self.updateHeatmap(samples, heat, heat_weights)

        self.p.subplot(1, 1)
        dargs = dict(color = 'tan', ambient=0.0, opacity=0.6, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
        self.gelsightActorB = self.p.add_mesh(transformed_gelsight_mesh, **dargs)

        # ground truth pose 
        # gt_quiver = pose2quiver(gt_pose, 5e-2)
        # gt_quiver.set_active_vectors("xvectors")
        # self.quiverXActor = self.p.add_mesh(gt_quiver.arrows, color="red", show_scalar_bar=False)
        # gt_quiver.set_active_vectors("yvectors")
        # self.quiverYActor = self.p.add_mesh(gt_quiver.arrows, color="green", show_scalar_bar=False)
        # gt_quiver.set_active_vectors("zvectors")
        # self.quiverZActor = self.p.add_mesh(gt_quiver.arrows, color="blue", show_scalar_bar=False)

        # visualize gelsight image 
        s = 0.002
        self.p.subplot(0, 1)
        imagetex = pv.numpy_to_texture(image)
        plane = pv.Plane(i_size = image.shape[1] * s, j_size = image.shape[0] * s, i_resolution = image.shape[1] - 1, j_resolution = image.shape[0] - 1)
        self.imageActor = self.p.add_mesh(plane, texture=imagetex, smooth_shading = True,  show_scalar_bar=False)
        plane.points[:, -1] = 0.2

        imagetex = pv.numpy_to_texture(-heightmap * mask.astype(np.float32))
        plane = pv.Plane(i_size = image.shape[1] * s, j_size = image.shape[0] * s, i_resolution = image.shape[1] - 1, j_resolution = image.shape[0] - 1)
        plane.points[:, -1] = np.flip(heightmap * mask.astype(np.float32), axis = 0).ravel() * (0.5*s) - 0.2
        plasma = cm.get_cmap('plasma')
        self.heightmapActor = self.p.add_mesh(plane, texture=imagetex, cmap = plasma, show_scalar_bar=False)

        self.frame_text = self.p.add_text(f'\nFrame {count}   ', position='upper_right', color='black', shadow=True, font = 'times', font_size=15)
                            
        if image_savepath:
            self.p.screenshot(image_savepath)  

        self.p.write_frame()  # write

    def drawGraph(self, x, y, savepath, delay, flag = 't'):

        if self.virtual_buff: 
            return
        
        fig, ax = plt.subplots()

        plt.xlabel('Timestep', fontsize=12)
        if flag == 't':
            plt.ylabel('Avg. translation RMSE (mm)', fontsize=12)
            y = [y_ * 1000.0 for y_ in y]
        elif flag == 'r':
            plt.ylabel('Avg. rotation RMSE (deg)', fontsize=12)

        # rolling avg. over 10 timesteps
        import pandas as pd
        df = pd.DataFrame()
        N = 50
        df['y'] = y
        df_smooth = df.rolling(N).mean()
        df_smooth['y'][0:N - 1] = y[0:N-1] # first 10 readings are as-is
        y = df_smooth['y']
        
        N, maxy = len(x), max(y)
        line, = ax.plot(x, y, color='k')

        def update(num, x, y, line):
            line.set_data(x[:num], y[:num])
            line.axes.axis([0, N, 0, maxy])
            return line,

        ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line], interval = delay, blit=True)
        ani.save(savepath + ".mp4", writer='ffmpeg', codec='h264')
        fig.savefig(savepath + ".pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)

    def closeDensityMesh(self):
        if self.virtual_buff: 
            return
        self.p.close()
        pv.close_all()

    def saveTSNEMesh(self, samples, clusters, save_path, nPoints = 500, radius_factor = 80.0):
        samples = np.atleast_2d(samples)
        samplePoints = pv.PolyData(samples[:, :3])
        samplePoints["similarity"] = clusters
        
        mesh = self.mesh.interpolate(samplePoints, strategy="mask_points", radius=self.mesh.length/radius_factor)
        p = pv.Plotter(off_screen=self.off_screen, window_size=[1000, 1000])

        # replace black with gray 
        if clusters.ndim == 2:
            null_idx = np.all(mesh["similarity"]==np.array([0.0, 0.0, 0.0]),axis=1)
            mesh["similarity"][null_idx, :] = np.array([189/256, 189/256, 189/256])
            
        # Open a gif
        if clusters.ndim == 2:
            dargs = dict(scalars='similarity', rgb=True, interpolate_before_map = False, opacity=1, smooth_shading=True, show_scalar_bar=False,  silhouette=True)
        else:
            dargs = dict(scalars='similarity', cmap = cm.get_cmap('plasma'), interpolate_before_map = False, opacity=1, smooth_shading=True, show_scalar_bar=False,  silhouette=True)
        p.add_mesh(mesh, **dargs)

        if nPoints is not None:
            p.show(screenshot=save_path, auto_close = not self.off_screen)
            viewup = [0.5, 0.5, 1]
            path = p.generate_orbital_path(factor=8.0, viewup=viewup, n_points=nPoints,  shift=mesh.length/(np.sqrt(3)))
            p.open_movie(save_path + ".mp4")
            p.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.01, progress_bar = True)
        else:
            p.show(screenshot=save_path)
        p.close()
        pv.close_all()

    def loadDepthMap(self, depthFolder, idx):
        depthFiles = sorted(os.listdir(depthFolder), key=lambda y: int(y.split(".")[0]))
        return pv.read(os.path.join(depthFolder, depthFiles[idx])).points

    def loadDepthMaps(self, depthFolder, downsample = 5e-4):
        depthMaps = []
        depthFiles = sorted(os.listdir(depthFolder), key=lambda y: int(y.split(".")[0]))
        for depthFile in depthFiles: 
            pc = o3d.io.read_point_cloud(os.path.join(depthFolder, depthFile))
            # pc = pc.voxel_down_sample(downsample)
            pc = np.array(pc.points)
            # pc = pc[np.random.choice(pc.shape[0], 1000), :]
            # assert pc.shape[0] == 1000, 'Expected shape: {} Actual: {}'.format(1000, pc.shape[0])
            depthMaps.append(pc)
        return depthMaps

    # TODO: RGB BGR consistency
    def loadImage(self, imageFolder, idx):
        imageFiles = sorted(os.listdir(imageFolder), key=lambda y: int(y.split(".")[0]))
        # return cv2.imread(os.path.join(imageFolder, imageFiles[idx])) 
        return Image.open(os.path.join(imageFolder, imageFiles[idx])) 

    def loadImages(self, imageFolder):
        try:
            imageFiles = sorted(os.listdir(imageFolder), key=lambda y: int(y.split(".")[0]))
        except:
            imageFiles = sorted(os.listdir(imageFolder))
        images = []
        for imageFile in imageFiles: 
            if imageFile.endswith('.mp4'):
                continue

            im = Image.open(os.path.join(imageFolder, imageFile)) 
            images.append(np.array(im))
        return images

    def saveDepthMaps(self, pointclouds, save_path):
        for i, pointcloud in enumerate(pointclouds):
            pc = pv.PolyData(pointcloud)
            pc.save('{path}/{p_i}.ply'.format(path = save_path, p_i=i))  
        return

    def saveImage(self, tactileImage, i, save_path):
        tactileImage = Image.fromarray(tactileImage.astype('uint8'), 'RGB')
        tactileImage.save("{path}/{p_i}.jpg".format(path=save_path, p_i=i))

    '''Save tactile images as float 32 RGB images'''
    def saveImages(self, tactileImages, save_path):
        for i, tactileImage in enumerate(tactileImages):
            self.saveImage(tactileImage, i, save_path)
        return

    def saveHeightmap(self, heightmap, i, save_path):
        cv2.imwrite("{path}/{p_i}.jpg".format(path=save_path, p_i=i), heightmap.astype('float64')) 

    '''Save heightmaps as float 64 images'''
    def saveHeightmaps(self, heightmaps, save_path):
        for i, heightmap in enumerate(heightmaps):
            self.saveHeightmap(heightmap, i, save_path)
        return

    def saveContactMask(self, contactMask, i, save_path):
        cv2.imwrite("{path}/{p_i}.jpg".format(path=save_path, p_i=i), 255*contactMask.astype("uint8"))

    '''Save contact masks as boolean images'''
    def saveContactMasks(self, contactMasks, save_path):
        for i, contactMask in enumerate(contactMasks):
            self.saveContactMask(contactMask, i, save_path)
        return

    def vizSE3Grid(self, se3_grid, voxels, save_path = None):
        quivers = pose2quiver(se3_grid, self.mesh.length/50)
        p = pv.Plotter(off_screen=self.off_screen, window_size=[5000, 5000])

        # plot voxels with se(3) components 
        p.add_mesh(voxels, color = None, show_edges=True, opacity=0.5)
        dargs = dict(color="grey", ambient=0.6, opacity=0.8, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
        p.add_mesh(self.mesh, **dargs)

        quivers.set_active_vectors("zvectors")
        p.add_mesh(quivers.arrows, color="blue", show_scalar_bar=False)

        if save_path:
            p.show(screenshot=save_path)
        else:
            p.show(auto_close= not self.off_screen)
        p.close()
        pv.close_all()

    def animate(self, poses, heightmaps, pointclouds_world, images, contactmasks, save_path = None):
        poses = np.vstack(poses)
        # plot 2: mesh and arrows 
        pv.global_theme.multi_rendering_splitting_position = 0.7

        p = pv.Plotter(shape='1|4', border_color = "white", off_screen=self.off_screen, window_size=[1920, 1200])
        p.show(auto_close = not self.off_screen)
        p.open_movie(save_path + ".mp4", framerate=30)
        print("Animating measurement path at {}".format(save_path + ".mp4"))
        
        # _ = p.add_axes(box=True)

        # mesh with measurements 
        p.subplot(0)
        dargs = dict(color = "grey", ambient=0.6, opacity=0.8, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)

        p.add_mesh(self.mesh, **dargs)
        p.set_focus(self.mesh.center)
        p.camera.Zoom(3)
        p.camera_set = True

        # generate the trajectory spline
        spline = pv.Spline(poses[:, :3], poses[:, :3].shape[0])
        p.add_mesh(spline, line_width=2, color="k")

        pbar = tqdm(total=poses.shape[0])
        for i, (pose, heightmap, pointcloud_world) in enumerate(zip(poses, heightmaps, pointclouds_world)):
            # visualize pointcloud
            p.subplot(0)
            if pointcloud_world.shape[0]//10 == 0:
                continue
            downpcd =  pointcloud_world[np.random.choice(pointcloud_world.shape[0], pointcloud_world.shape[0]//10, replace=False), :]
            pc = pv.PolyData(downpcd)
            p.add_points(pc, render_points_as_spheres=True, color = "#26D701", point_size=3)

            # visualize gelsight 
            sensor_transform = gen_quat_t(pose)
            transformed_gelsight_mesh = self.gelsight_mesh.transform(sensor_transform, inplace = False)
            dargs = dict(color = "black", ambient=0.6, opacity=0.3, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
            gelsightActor = p.add_mesh(transformed_gelsight_mesh, **dargs)

            # visualize gelsight image 
            p.subplot(1)
            if i == 0:
                p.camera.Zoom(3)
            image = images[i]
            imagetex = pv.numpy_to_texture(image)
            plane = pv.Plane(i_size = image.shape[1] * 0.001, j_size = image.shape[0] * 0.001, i_resolution = image.shape[1], j_resolution = image.shape[0])
            imageActor = p.add_mesh(plane, texture=imagetex)

            # visualize heightmaps
            p.subplot(2)
            if i == 0:
                p.camera.Zoom(3)
            image = -heightmap * contactmasks[i].astype(np.float32)
            imagetex = pv.numpy_to_texture(image)
            viridis = cm.get_cmap('viridis')
            heightmapActor = p.add_mesh(plane, texture=imagetex, cmap = viridis)

            p.write_frame()  # write initial data
            p.remove_actor(gelsightActor)
            p.remove_actor(heightmapActor)
            p.remove_actor(imageActor)
            pbar.update(1)
        pbar.close()
        p.close()
        pv.close_all()

    def vizPoses(self, query_pose, target_poses):
        p = pv.Plotter(window_size=[2000, 2000])
        quivers = pose2quiver(target_poses, self.mesh.length/50) # compensate for the pen depth

        quivers.set_active_vectors("xvectors")
        dargs = dict(color="red", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        quivers.set_active_vectors("yvectors")
        dargs = dict(color="green", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        quivers.set_active_vectors("zvectors")
        dargs = dict(color="blue", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)


        quivers = pose2quiver(query_pose, self.mesh.length/25) # compensate for the pen depth

        quivers.set_active_vectors("xvectors")
        dargs = dict(color="red", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        quivers.set_active_vectors("yvectors")
        dargs = dict(color="green", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        quivers.set_active_vectors("zvectors")
        dargs = dict(color="blue", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)

        dargs = dict(color="grey", ambient=0.6, opacity=0.6, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
        p.add_mesh(self.mesh, **dargs)

        p.show()
        p.close()
        
        pv.close_all()


    def vizMeasurements(self, poses, pointclouds, save_path = None, decimation_factor = 5):
        if type(pointclouds) is not list:
            temp = pointclouds
            pointclouds = [None] * 1
            pointclouds[0] = temp
        # quivers = pose2quiver(poses, self.mesh.length/50)      

        quivers = pose2quiver(poses, self.mesh.length/50) # compensate for the pen depth
        # generate the trajectory spline

        # plot 2: mesh and arrows 
        p = pv.Plotter(off_screen=self.off_screen, window_size=[2000, 2000])
        quivers.set_active_vectors("xvectors")
        dargs = dict(color="red", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        quivers.set_active_vectors("yvectors")
        dargs = dict(color="green", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        quivers.set_active_vectors("zvectors")
        dargs = dict(color="blue", show_scalar_bar=False)
        p.add_mesh(quivers.arrows, **dargs)
        dargs = dict(color="grey", ambient=0.6, opacity=0.6, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
        p.add_mesh(self.mesh, **dargs)
        # plot without scalars

        spline = pv.lines_from_points(poses[:, :3])
        p.add_mesh(spline, line_width=3, color="k")

        final_pc = np.empty((0, 3))
        for i, pointcloud in enumerate(pointclouds):
            if pointcloud.shape[0] == 0:
                continue
            if decimation_factor is not None: 
                downpcd =  pointcloud[np.random.choice(pointcloud.shape[0], pointcloud.shape[0]//decimation_factor, replace=False), :]
            else:
                downpcd = pointcloud
            final_pc = np.append(final_pc, downpcd)
        
        if final_pc.shape[0]:
            pc = pv.PolyData(final_pc)
            p.add_points(pc, render_points_as_spheres=True, color = "#26D701", point_size=3)
    
        if save_path:
            p.show(screenshot=save_path)
            print(f"Save path: {save_path}.png")
        else:
            p.show(auto_close= not self.off_screen)
        p.close()
        pv.close_all()

    def vizAnnotations(self, annotations, save_path = None):
            # plot 2: mesh and arrows 
            p = pv.Plotter(off_screen=self.off_screen, window_size=[5000, 5000])

            dargs = dict(color="grey", ambient=0.6, opacity=0.6, smooth_shading=True, show_edges=False, specular=1.0, show_scalar_bar=False)
            p.add_mesh(self.mesh, **dargs)

            colors = {'flats': np.array([131, 166, 199]), 'edges': np.array([250, 136, 197]), 'corners': np.array([255, 166, 0])}
            # visualize corners, edges, faces
            for ann, points in annotations.items():
                pc = pv.PolyData(self.mesh.points[points])
                # create many spheres from the point cloud
                pc['colors'] = np.repeat(np.atleast_2d(colors[ann]), repeats=len(points), axis=0)
                p.add_points(pc, render_points_as_spheres=True, point_size=5, scalars='colors', rgb=True)

            if save_path:
                p.show(screenshot=save_path)
            else:
                p.show(auto_close = not self.off_screen)
            p.close()
            pv.close_all()

    def computeNeighborhood(self, points, rad = 1e-3):
        tree = scipy.spatial.KDTree(self.mesh.points)
        query_idxs = tree.query_ball_point(points, r=rad)

        combined_idxs = []
        for query_idx in query_idxs:
            combined_idxs= list(set(combined_idxs) | set(query_idx))

        return self.mesh.points[combined_idxs]

    def all_vertices(self):
        return self.mesh.points

    def pick_points(self):
        '''http://www.open3d.org/docs/latest/tutorial/visualization/interactive_visualization.html'''
        print("")
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.vertices)

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        return np.asarray(pcd.points)[vis.get_picked_points(), :]
        # return vis.get_picked_points()

    def getGeodesicPath(self, start_point, end_point):
        start_point_idx = np.argmin(np.linalg.norm(start_point - self.mesh.points, axis=1))
        end_point_idx = np.argmin(np.linalg.norm(end_point - self.mesh.points, axis=1))
        path_pts = self.mesh.geodesic(start_point_idx, end_point_idx) # shares precomputation for repeated solves
        path_distance = self.mesh.geodesic_distance(start_point_idx, end_point_idx)

        # DEBUG: plot geodesic path         
        # p = pv.Plotter()
        # p.add_mesh(path_pts, line_width=10, color="red", label="Geodesic Path")
        # p.add_mesh(self.mesh, show_edges=True)
        # p.show()
        return path_pts.points, path_distance

    def getShortestPath(self, start_point, end_point):
        '''shortest.ipynb
        Given a mesh and two vertex indices find the shortest path
        between the two vertices while only traveling along edges
        of the mesh.'''
        import trimesh
        import networkx as nx
        # test on a sphere mesh
        mesh = trimesh.load(self.mesh_path)
        # edges without duplication
        edges = mesh.edges_unique
        # the actual length of each unique edge
        length = mesh.edges_unique_length
        # create the graph with edge attributes for length
        g = nx.Graph()
        for edge, L in zip(edges, length):
            g.add_edge(*edge, length=L)

        start_idx = np.argmin(np.linalg.norm(start_point - mesh.vertices, axis=1))
        end_idx = np.argmin(np.linalg.norm(end_point - mesh.vertices, axis=1))

        # run the shortest path query using length for edge weight
        path = nx.shortest_path(g,
                                source=start_idx,
                                target=end_idx,
                                weight='length')
        return mesh.vertices[path]
        
    def subsamplePath(self, samplePath, sz = 100):
        samplePath, _ = np.unique(samplePath, axis=0, return_index=True) # remove duplicates 
        assert(samplePath.shape[0] > sz), "{p} poses and {s} samples".format(p = samplePath.shape[0], s = sz)
        samples = samplePath[::(samplePath.shape[0]//sz)]

        delta = sz - samples.shape[0]
        print(delta)
        if delta < 0:
            idx = np.random.choice(samples.shape[0], size =  samples.shape[0] + delta, replace=False) # remove 
            samples = samples[np.sort(idx), :]

        assert(samples.shape[0] == sz), "sampling not correct"
        return samples 

def pose2quiver(poses, sz):
    poses = np.atleast_2d(poses)
    quivers = pv.PolyData(poses[:,:3]) # (N, 3) [x, y, z]
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r = R.from_quat(poses[:,3:]) # (N, 4) [qx, qy, qz, qw]
    quivers["xvectors"]  = r.apply(x) * sz
    quivers["yvectors"]  = r.apply(y) * sz
    quivers["zvectors"]  = r.apply(z) * sz
    quivers.set_active_vectors("zvectors")
    return quivers

def loadHeightmapAndMask(heightmapFile, maskFile):
    try:
        heightmap = cv2.imread(heightmapFile, 0).astype(np.int64)
        contactmask = cv2.imread(maskFile, 0).astype(np.int64)
    except AttributeError: 
        heightmap = np.zeros(heightmap.shape).astype(np.int64)
        contactmask = np.zeros(contactmask.shape).astype(np.int64)
    return heightmap, contactmask > 255/2

def loadHeightmapsAndMasks(heightmapFolder, contactmaskFolder):
    heightmapFiles = sorted(os.listdir(heightmapFolder), key=lambda y: int(y.split("_")[0]))
    contactmaskFiles = sorted(os.listdir(contactmaskFolder), key=lambda y: int(y.split("_")[0]))
    heightmaps, contactmasks = [], []

    for heightmapFile, contactmaskFile in zip(heightmapFiles, contactmaskFiles): 
        heightmap, mask = loadHeightmapAndMask(os.path.join(heightmapFolder, heightmapFile), os.path.join(contactmaskFolder, contactmaskFile))
        heightmaps.append(heightmap)
        contactmasks.append(mask)
    return heightmaps, contactmasks
