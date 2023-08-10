import numpy as np
import cv2
import os
import pickle, lzma

def step(uid):
    for yaw in np.arange(0, 346, 15):
        for pitch in np.arange(-15, 46, 15):
            name = '{}_{}'.format(yaw, pitch)
            depth_path = os.path.join('datasets/shapenet/pytorch3d_render/depth/', uid, name+'_0001.lzma')
            normal_path = os.path.join('datasets/shapenet/pytorch3d_render/normal/', uid, name+'_0001.lzma')
            sketch_path = os.path.join('datasets/shapenet/pytorch3d_render/sketch/', uid, name+'_0001.png')
            sketch_trans_path = os.path.join('datasets/shapenet/pytorch3d_render/sketch_transparent/', uid, name+'_0001.png')
            if os.path.exists(depth_path) and os.path.exists(normal_path) and not os.path.exists(sketch_path):
                os.makedirs(os.path.join('datasets/shapenet/pytorch3d_render/sketch_transparent/', uid), exist_ok=True)
                os.makedirs(os.path.join('datasets/shapenet/pytorch3d_render/sketch/', uid), exist_ok=True)
                print(depth_path)
                float_depth = pickle.load(lzma.open(depth_path, 'rb'))
                float_normal = pickle.load(lzma.open(normal_path, 'rb'))
                depth = np.array(float_depth*127.5, dtype=np.uint8)
                normal = np.array((float_normal+1)*127.5, dtype=np.uint8)


                normal_sketch = normal[:,:,2]
                normal_sketch = cv2.GaussianBlur(normal_sketch,(3,3),0)
                normal_sketch = cv2.adaptiveThreshold(normal_sketch, 
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    blockSize=3,
                                    C=3)
                depth_sketch = depth
                depth_sketch = cv2.adaptiveThreshold(depth_sketch, 
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        blockSize=5,
                                        C=-1)

                sketch = cv2.bitwise_and(normal_sketch, 255-depth_sketch)
                cv2.imwrite(sketch_path, sketch)

                sketch_trans = np.zeros((256,256,4),dtype=np.uint8)
                sketch_trans[:,:,3] = 255-sketch
                cv2.imwrite(sketch_trans_path, sketch_trans)
                



if __name__ == '__main__':
    uids = np.loadtxt('datasets/shapenet/splits/all_02691156.lst', dtype=str)
    for uid in uids:
        step(uid)

      





