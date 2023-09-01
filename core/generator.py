import mcubes
import trimesh
import torch
from torch.nn import functional as F
import os
from glob import glob
import numpy as np
from core.geometry import create_grid_points_from_bounds

class Generator(object):
    def __init__(self, model, threshold,
                 exp_name, checkpoint=None, device=torch.device("cuda"), resolution=128, num_views=3):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format(exp_name)
        self.load_checkpoint(checkpoint)

        self.min = -0.5
        self.max = 0.5
        self.batch_points = 65536
        self.resolution = resolution
        self.threshold = threshold
        self.num_views = num_views

        points = create_grid_points_from_bounds(self.min, self.max, self.resolution)
        self.points = torch.from_numpy(points).float()
        self.points_split = torch.split(self.points.reshape(1, -1, 3), self.batch_points, dim=1)  

    def generate_mesh(self, data):
        img0 = data['img0'].to(self.device)
        calib0 = data['calib0'].to(self.device)
        if self.num_views == 1:
            with torch.no_grad():
                logits = self.model.forward_with_split(self.points_split, img0, calib0, self.device)
        else:
            img_lst = [img0]
            calib_lst = [calib0]
        
            img1 = data['img1'].to(self.device)
            calib1 = data['calib1'].to(self.device)
            img_lst.append(img1)
            calib_lst.append(calib1)

            if self.num_views > 2: # 3
                img2 = data['img2'].to(self.device)
                calib2 = data['calib2'].to(self.device)
                img_lst.append(img2)
                calib_lst.append(calib2)

            with torch.no_grad():
                logits = self.model.forward_with_split(self.points_split, img_lst, calib_lst, self.num_views, self.device)
        return logits


    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)
        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, epoch):
        if epoch is None:
            epochs = glob(self.checkpoint_path+'/*')
            if len(epochs) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            epochs = [os.path.splitext(os.path.basename(path))[0][17:] for path in epochs]
            epochs = np.array(epochs, dtype=int)
            epochs = np.sort(epochs)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epochs[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

