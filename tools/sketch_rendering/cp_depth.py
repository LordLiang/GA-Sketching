import numpy as np
import os
import pickle, lzma

def step(uid):
    for yaw in np.arange(0, 346, 15):
        for pitch in np.arange(-15, 46, 15):
            old_name = '{}_{}'.format((360-yaw)%360, pitch)
            name = '{}_{}'.format(yaw, pitch)
            old_depth_path = os.path.join('datasets/shapenet1/pytorch3d_render/depth/', uid, old_name+'_0001.lzma')
            depth_path = os.path.join('datasets/shapenet/pytorch3d_render/depth/', uid, name+'_0001.lzma')
            if os.path.exists(old_depth_path) and not os.path.exists(depth_path):
                os.makedirs(os.path.join('datasets/shapenet/pytorch3d_render/depth/', uid), exist_ok=True)
                os.system('cp ' + old_depth_path + ' ' + depth_path)

if __name__ == '__main__':
    uids = np.loadtxt('datasets/shapenet/splits/all_03001627.lst', dtype=str)
    for uid in uids:
        step(uid)

      





