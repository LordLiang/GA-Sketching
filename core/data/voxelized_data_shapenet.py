import os
import numpy as np
import cv2
from scipy import ndimage
import pickle, lzma

import torch
from torch.utils.data import Dataset

from core.models.s2d_model import Pix2pixNet

def make_rotate(rx, ry, rz):
    sinX, cosX = np.sin(rx), np.cos(rx)
    sinY, cosY = np.sin(ry), np.cos(ry)
    sinZ, cosZ = np.sin(rz), np.cos(rz)
    Rx = np.array([[1.0,    0,     0],
                   [  0, cosX, -sinX],
                   [  0, sinX,  cosX]])

    Ry = np.array([[ cosY,   0, sinY],
                   [    0, 1.0,    0],
                   [-sinY,   0, cosY]])

    Rz = np.array([[cosZ, -sinZ,   0],
                   [sinZ,  cosZ,   0],
                   [   0,     0, 1.0]])
    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def get_extrinsic(az, el):
    extrinsic = np.identity(4)
    R = np.matmul(make_rotate(np.radians(el), 0, 0), make_rotate(0, np.radians(-az-90), 0))
    extrinsic[0:3, 0:3] = R
    return extrinsic

kernel = np.ones((3, 3), dtype=np.uint8)
def gauss_blur(image):
    mask = ((image == 0)*255).astype(np.uint8)
    mask = cv2.dilate(mask, kernel, 1)
    mask = (mask==255)

    out = (image * 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = out.astype(np.float32)/255
    out = np.clip(out, 0.0, 1.0)
    out[mask] = image[mask]
    return out

def jitter_depth(depth, sigma_1=0.02, sigma_2=0.02):
    mask = (depth == 0)
    # local jitter
    jittered_depth = sigma_1 * np.random.randn(8, 8)
    jittered_depth = ndimage.zoom(jittered_depth, (32, 32), order=3)
    out = depth + jittered_depth
    # global jitter
    out = out + sigma_2 * np.random.randn()
    out = np.clip(out, 0.0, 1.0)
    out[mask] = 0
    return out.astype(np.float32)

category_id = {'chair': '03001627',
               'airplane': '02691156'}

class VoxelizedDataset(Dataset):

    def __init__(self, mode, category,
                 batch_size=4, num_sample_points=50000, num_workers=12, 
                 sample_distribution=[0.5, 0.5], sample_sigmas=[0.2, 0.015]):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)
    
        self.mode = mode
        self.data_dir = 'datasets/shapenet/'
        self.category_id = category_id[category]
        self.data = np.loadtxt('datasets/shapenet/splits/{}_{}.lst'.format(mode, self.category_id), dtype=str)
        
        if self.mode == 'test':
            self.dict = np.load('{}/splits/multi-view_test_{}.npy'.format(self.data_dir, self.category_id), allow_pickle=True).item()
            if category == 'chair':
                ck_path_s2d = 'pretrained/s2d/chair/200_net_G-c.pth'.format(category)
            elif category == 'airplane':
                ck_path_s2d = 'pretrained/s2d/airplane/200_net_G-b.pth'.format(category)
            else:
                print('Not Supported.')
                quit()
            # s2d
            self.s2d_net = Pix2pixNet(1, 2, 64).cuda()
            self.s2d_net.eval()
            checkpoint = torch.load(ck_path_s2d)
            self.s2d_net.netG.load_state_dict(checkpoint)

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)

        # for normal map
        self.loadSize = 256
        self.az_list = np.arange(0, 346, 15)
        self.el_list = np.arange(-15, 46, 15)
        
        # Match image pixel space to image uv space
        self.uv_intrinsic = np.identity(4)
        self.uv_intrinsic[0, 0] = 1.0 / float(self.loadSize // 2)
        self.uv_intrinsic[1, 1] = 1.0 / float(self.loadSize // 2)
        self.uv_intrinsic[2, 2] = 1.0 / float(self.loadSize // 2)

        self.extrinsic_list = {}
        for az in self.az_list:
            for el in self.el_list:
                name = '{}_{}'.format(az, el)
                self.extrinsic_list[name] = get_extrinsic(az, el)

        self.morph_kernel = np.ones((2, 2), np.uint8)
        self.view_code = np.array([-1, 0, 0])

    def __len__(self):
        return len(self.data)

    def get_img_calib(self, uid, yaw, pitch):
        name = '{}_{}'.format(yaw, pitch)
        extrinsic = self.extrinsic_list[name]

        sketch_path =  os.path.join(self.data_dir, 'pytorch3d_render/sketch', uid, name + '_0001.png')
        sketch = 255 - cv2.imread(sketch_path, 0)

        if self.mode != 'test':
            depth_path =  os.path.join(self.data_dir, 'pytorch3d_render/depth', uid, name + '_0001.lzma')
            depth = 1-pickle.load(lzma.open(depth_path, 'rb'))*0.5 # [0,1]


        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 256
        scale_intrinsic[1, 1] = -256
        scale_intrinsic[2, 2] = 256
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)

        # augmentation
        if self.mode == 'train':
            # Pad images
            pad_size = int(0.1 * self.loadSize)
            sketch = np.pad(sketch, pad_size, 'constant', constant_values=(0))
            depth = np.pad(depth, pad_size, 'constant', constant_values=(0))
            s = depth.shape[0]
            # random translate
            dx = np.random.randint(-int(round((s - self.loadSize) / 10.)),
                                    int(round((s - self.loadSize) / 10.)))
            dy = np.random.randint(-int(round((s - self.loadSize) / 10.)),
                                    int(round((s - self.loadSize) / 10.)))

            trans_intrinsic[0, 3] = -dx / float(self.loadSize // 2)
            trans_intrinsic[1, 3] = -dy / float(self.loadSize // 2)
            x1 = int(round((s - self.loadSize) / 2.)) + dx
            y1 = int(round((s - self.loadSize) / 2.)) + dy
            sketch = sketch[y1:y1+self.loadSize, x1:x1+self.loadSize]
            depth = depth[y1:y1+self.loadSize, x1:x1+self.loadSize]


        intrinsic = np.matmul(trans_intrinsic, np.matmul(self.uv_intrinsic, scale_intrinsic))
        calib = np.matmul(intrinsic, extrinsic).astype(np.float32)


        # morph augmentation
        if self.mode == 'train':
            val = np.random.rand()
            if val < 0.2:
                #erode
                sketch = cv2.erode(sketch, self.morph_kernel)
            elif val < 0.4:
                #dilate
                sketch = cv2.dilate(sketch, self.morph_kernel)
            elif val < 0.6:
                sketch = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, self.morph_kernel)
            elif val < 0.8:
                sketch = cv2.morphologyEx(sketch, cv2.MORPH_CLOSE, self.morph_kernel)
            else:
                pass

        # add gauss blur and jitter
        if self.mode == 'train':
            if np.random.rand() > 0.5:
                depth = gauss_blur(depth)
            if np.random.rand() > 0.5:
                depth = jitter_depth(depth) 

        sketch = sketch.astype(np.float32)/255.
        if self.mode == 'test':
            R = make_rotate(np.radians(-pitch), np.radians(90+yaw), 0)
            view_code = np.matmul(self.view_code, R).astype(np.float32)
            sketch_tensor = torch.from_numpy(sketch).unsqueeze(0).unsqueeze(0).cuda()
            view_code_tensor = torch.from_numpy(view_code).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            depth_tensor = self.s2d_net(sketch_tensor, view_code_tensor)
            depth = depth_tensor[0][0].detach().cpu().numpy()

        img = np.stack((sketch, depth), 0)
        return img, calib

    def __getitem__(self, idx):
        uid = self.data[idx]
        ret = {'uid' : uid}

        if self.mode == 'train':
            azs = np.random.choice(self.az_list, 3, replace=True)
            els = np.random.choice(self.el_list, 3, replace=True)
        else:
            azs = [45, 165, 285]
            els = [30, 30, 30]
            if self.mode == 'test':
                a0, e0, a1, e1, a2, e2 = self.dict[uid]
                azs = [a0, a1, a2]
                els = [e0, e1, e2]
           
        ret['img0'], ret['calib0'] = self.get_img_calib(uid, azs[0], els[0])
        ret['img1'], ret['calib1'] = self.get_img_calib(uid, azs[1], els[1])
        ret['img2'], ret['calib2'] = self.get_img_calib(uid, azs[2], els[2])
        
        if self.mode in ['train', 'val']:
            points = []
            occupancies = []
            for i, num in enumerate(self.num_samples):
                boundary_samples_path = os.path.join(self.data_dir, 'boundary_sampling', uid, 'boundary_{}_samples.npz'.format(self.sample_sigmas[i]))
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_occupancies = boundary_samples_npz['occupancies']
                subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
                points.extend(boundary_sample_points[subsample_indices])
                occupancies.extend(boundary_sample_occupancies[subsample_indices])

            ret['occupancies'] = np.array(occupancies, dtype=np.float32)
            ret['points'] = np.array(points, dtype=np.float32)

        return ret

    def get_loader(self, shuffle =True):
        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

