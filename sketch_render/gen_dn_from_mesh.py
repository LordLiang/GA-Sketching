import torch
import trimesh
import numpy as np
from pytorch3d.renderer.cameras import (
    look_at_view_transform,
    OrthographicCameras,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
)
import pickle, lzma
import os, glob

class Renderer(torch.nn.Module):
    def __init__(self, depth_renderer, image_size=256):
        super().__init__()
        self.depth_renderer = depth_renderer

        # sobel filters
        with torch.no_grad():
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        # Pixel coordinates
        self.X, self.Y = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size))
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().cuda()
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().cuda()

    def depth_2_normal(self, depth, depth_unvalid, cameras):

        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).cuda()
        # Pixel coordinates
        xy_depth = torch.cat([self.X, self.Y, depth], 3).cuda().reshape(B,-1, 3)
        xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

        # compute tangent vectors
        XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
        vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
        vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

        # finally compute cross product
        normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
        normal_norm = normal.norm(p=2, dim=1, keepdim=True)

        normal_normalized = normal.div(normal_norm)
        # reshape to image
        normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
        grad_out[:,1:-1,1:-1,:] = -normal_out

        # zero out +Inf
        grad_out[depth_unvalid] = 0.0

        return grad_out

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = 2
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras'])

        return depth_out, normals_out

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


IMG_SIZE = 256
R, T = look_at_view_transform(1, 0, 0)
device = torch.device("cuda:0")
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
cameras = OrthographicCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(
    image_size=IMG_SIZE,
    blur_radius=0.000001,
    faces_per_pixel=1,
)
depth_renderer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)
renderer = Renderer(depth_renderer, image_size=IMG_SIZE)

def step(uid):
    obj_path = os.path.join('datasets/shapenet/watertight_simplified_off', uid, 'watertight_simplified.off')
    if not os.path.exists(obj_path):
        return
    fns = glob.glob(os.path.join('datasets/shapenet/pytorch3d_render/depth', uid, '*.lzma'))
    fns2 = glob.glob(os.path.join('datasets/shapenet/pytorch3d_render/normal', uid, '*.lzma'))
    if len(fns)<24*5 or len(fns2)<24*5:
        print(uid, len(fns), len(fns2))
        mesh = trimesh.load(obj_path)
        for yaw in np.arange(0, 346, 15):
            for pitch in np.arange(-15, 46, 15):#you can change it to -90~90 if you want
                name = '{}_{}'.format(yaw, pitch)
                depth_path = os.path.join('datasets/shapenet/pytorch3d_render/depth', uid, name + '_0001.lzma')
                normal_path = os.path.join('datasets/shapenet/pytorch3d_render/normal', uid, name + '_0001.lzma')
                if (not os.path.exists(depth_path)) or (not os.path.exists(normal_path)):
                    os.makedirs(os.path.join('datasets/shapenet/pytorch3d_render/depth', uid), exist_ok=True)
                    os.makedirs(os.path.join('datasets/shapenet/pytorch3d_render/normal', uid), exist_ok=True)
                    R = make_rotate(np.radians(-pitch), np.radians(90+yaw), 0)
                    pc = np.matmul(mesh.vertices.transpose(0, 1), R).transpose(0, 1)
                    pc = np.array(pc*2, dtype=np.float32)
                    verts = torch.from_numpy(pc).to(device)
                    faces = torch.from_numpy(mesh.faces).float().to(device)
                    verts = verts.unsqueeze(0)
                    faces = faces.unsqueeze(0)
                    meshes = Meshes(verts, faces)
                    depth_out, normal_out = renderer(meshes_world=meshes, cameras=cameras, lights=lights)
                    depth = depth_out.detach().cpu().numpy()[0]
                    normal = normal_out.detach().cpu().numpy()[0]
                    pickle.dump(depth, lzma.open(depth_path, 'wb'))
                    pickle.dump(normal, lzma.open(normal_path, 'wb'))
 

if __name__ == '__main__':
    uids = np.loadtxt('datasets/shapenet/splits/all_02691156.lst', dtype=str)
    for uid in uids:
        step(uid)



