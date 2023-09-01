from PyQt5.QtGui import QImage

import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from pytorch3d.renderer.cameras import (
    look_at_view_transform,
    OrthographicCameras,
)

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)

import trimesh
import open3d as o3d

from configs.config import cfg
from models.gas_model import GASNet_MV
from models.s2d_model import Pix2pixNet
from models.utils.pc_util import *
from models.utils.img_util import open, close, dilate, zoom_3d

category_map = ['chair', 'airplane']


class GeometryManager:
    def __init__(self):
        super(GeometryManager, self).__init__()
        self.input_res = 64
        self.output_res = 128
        self.batch_points = 65536
        self.image_size = 256
        self.device = torch.device("cuda:0")
        self.category_id = 0
        self.with_depth_updater = True # you should test this!!!
        self.axis_vector = np.array([-1, 0, 0])

        # s2d
        self.s2d_net = Pix2pixNet(1, 2, 64).to(self.device)
        self.s2d_net.eval()

        # s2d_update
        self.s2d_update_net = Pix2pixNet(2, 2, 64).to(self.device)
        self.s2d_update_net.eval()
        
        # gas
        self.gas_net = GASNet_MV(self.input_res).to(self.device)
        self.gas_net.eval()

        ######## all model loading
        self.load_checkpoint()
        
        # query for output
        points = create_grid_points_from_bounds(-0.5, 0.5, self.output_res).astype(np.float32)
        points = torch.from_numpy(points).unsqueeze(0)
        self.grid_points_split = torch.split(points, self.batch_points, dim=1)

        ######## renderer setting
        R, T = look_at_view_transform(1, 0, 0)
        self.cameras_pytorch3d = OrthographicCameras(device=self.device, R=R, T=T)
        depth_raster_settings = RasterizationSettings(
            image_size=self.image_size
        )
        self.depth_renderer = MeshRasterizer(
            cameras=self.cameras_pytorch3d,
            raster_settings=depth_raster_settings
        )

        self.X, self.Y = torch.meshgrid(torch.arange(0, self.image_size), torch.arange(0, self.image_size))
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/self.image_size - 1).float().to(self.device)
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/self.image_size - 1).float().to(self.device)

        self.vertices = None
        self.triangles = None
        self.count = -1
        self.use_normal = True
        self.coarse_volume = None
        self.zoom_factor = 1.0
        self.feature_zoom_factor = 1.0
        self.axis_vector = np.array([-1, 0, 0])

    def change_category(self, category_id):
        if category_id != self.category_id:
            self.category_id = category_id
            ######## all model loading
            self.load_checkpoint()
            
    def load_checkpoint(self):
        if self.category_id == 0:
            ck_path_s2d = 'pretrained/s2d/chair/200_net_G-c.pth'
            ck_path_s2d_uptate = 'pretrained/s2d_update/chair/200_net_G-b.pth'
            ck_path_gas = 'pretrained/single_view/chair/checkpoint_epoch_200.tar'
            ck_path_merger = 'pretrained/merger/chair/checkpoint_epoch_200.tar'

        elif self.category_id == 1:
            ck_path_s2d = 'pretrained/s2d/airplane/200_net_G-b.pth'
            ck_path_s2d_uptate = 'pretrained/s2d_update/airplane/200_net_G-b.pth'
            ck_path_gas = 'pretrained/single_view/airplane/checkpoint_epoch_200.tar'
            ck_path_merger = 'pretrained/merger/airplane/checkpoint_epoch_200.tar'
        else:
            print('Not Supported!')
            quit()
        self.s2d_net.netG.load_state_dict(torch.load(ck_path_s2d))
        self.s2d_update_net.netG.load_state_dict(torch.load(ck_path_s2d_uptate))
        self.gas_net.single_view_net.load_state_dict(torch.load(ck_path_gas))
        self.gas_net.merger.load_state_dict(torch.load(ck_path_merger))
        return
        

    def depth2normal(self, depth, depth_unvalid):
        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).cuda()
        # Pixel coordinates
        xy_depth = torch.cat([self.X, self.Y, depth], 3).cuda().reshape(B, -1, 3)
        xyz_unproj = self.cameras_pytorch3d.unproject_points(xy_depth, world_coordinates=False)

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


    def normal2sketch(self, float_depth, float_normal):
        depth = np.array(float_depth*127.5, dtype=np.uint8)
        normal = np.array((float_normal+1)*127.5, dtype=np.uint8)

        normal_sketch = cv2.GaussianBlur(normal[:,:,2],(3,3),0)
        #normal_sketch[float_depth==2] = 0
        normal_sketch = cv2.adaptiveThreshold(normal_sketch, 
                                255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                blockSize=3,
                                C=3)

        depth_sketch = cv2.adaptiveThreshold(depth, 
                                255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                blockSize=5,
                                C=-1)

        sketch = cv2.bitwise_and(normal_sketch, 255-depth_sketch)
        return 1-sketch/255


    def depth2sketch(self, float_depth):
        depth = np.array((2-float_depth)*127.5, dtype=np.uint8)
        sketch = cv2.adaptiveThreshold(depth, 
                                       255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       blockSize=5,
                                       C=3)
        return 1-sketch/255


    def render_sketch(self, R):
        if self.use_normal:
            # use both depth and normal t0 generate sketch
            depth_out, normal_out = self.render_depth(R, return_normal=True)
            sketch = self.normal2sketch(depth_out[0].detach().cpu().numpy(), normal_out[0].detach().cpu().numpy()).astype(np.uint8)
        else:
            # use depth to generate sketch
            depth_out = self.render_depth(R)
            sketch = self.depth2sketch(depth_out[0].detach().cpu().numpy()).astype(np.uint8)

        # morph_kernel = np.ones((2,2), np.uint8)
        # sketch = cv2.morphologyEx(sketch, cv2.MORPH_CLOSE, morph_kernel)
        # sketch = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, morph_kernel)
        return sketch
    
    def empty_canvas(self, fill_color=(0,0,0)):
        canvas = np.ones((256, 256, 4), dtype=np.uint8)*255
        canvas[:,:,0] = fill_color[0]
        canvas[:,:,1] = fill_color[1]
        canvas[:,:,2] = fill_color[2]
        return canvas
    
    def render_depth(self, R, return_normal=False):
        if self.vertices is None:
            self.generate_reconstructed_mesh_from_point()

        pc = np.matmul(self.vertices.transpose(0, 1), R).transpose(0, 1)
        pc = np.array(pc*2, dtype=np.float32)
        verts = torch.from_numpy(pc).to(self.device)
        faces = torch.from_numpy(self.triangles.astype(np.float32)).to(self.device)
        meshes = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
        depth_ref = self.depth_renderer(meshes)
        z_buf = depth_ref.zbuf
        depth_unvalid = z_buf<0
        z_buf[depth_unvalid] = 2
        depth_out = z_buf[..., 0]
        if return_normal:
            normal_out = self.depth2normal(z_buf, depth_unvalid.squeeze(-1))
            return depth_out, normal_out
        else:
            return depth_out
    
    def get_view_code_mid(self, R):
        view_code = np.matmul(self.axis_vector, R)
        view_code = torch.from_numpy(np.array(view_code, dtype=np.float32))
        view_code = view_code.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)
        return view_code


    def generate_sketch(self, azi, ele, save_dir=None):
        if self.vertices is None:
            self.generate_reconstructed_mesh_from_point()
        R = make_rotate(np.radians(-ele), np.radians(90+azi), 0)
        alpha = self.render_sketch(R)
        crease = self.empty_canvas()
        cv2.imwrite(os.path.join(save_dir, 'alpha_'+str(self.count)+'.png'), ((1-alpha)*255).astype(np.uint8))
        crease[:,:,3] *= alpha
        
        crease = cv2.resize(crease, (600, 600), interpolation=cv2.INTER_NEAREST)
        img_qt = QImage(crease.data, 600, 600, QImage.Format_ARGB32)
        return img_qt

    def generate_reconstructed_mesh(self, azi, ele, sketch, edit_mask, save_dir=None):
        self.count += 1
        print(self.count, azi, ele)
        R = make_rotate(np.radians(-ele), np.radians(90+azi), 0)
        view_code = self.get_view_code_mid(R)
        calib = get_calib(azi, ele)
        calib = torch.from_numpy(calib[np.newaxis,:].astype(np.float32)).to(self.device)
        cv2.imwrite(os.path.join(save_dir, 'sketch_'+str(self.count)+'.png'), ((1-sketch[0])*255).astype(np.uint8))
        
        # step1: sketch2depth
        sketch = torch.from_numpy(sketch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.s2d_net(sketch, view_code)
            depth_pil = Image.fromarray(((1-depth[0][0].detach().cpu().numpy())*255).astype(np.uint8))
            depth_pil.save(os.path.join(save_dir, 'depth_coarse_'+str(self.count)+'.png'))

        if self.count > 0 and self.with_depth_updater:
            depth_ref = (1-self.render_depth(R)*0.5).unsqueeze(0)
            depth_ref_pil = Image.fromarray(((1-depth_ref[0][0].detach().cpu().numpy())*255).astype(np.uint8))
            depth_ref_pil.save(os.path.join(save_dir, 'depth_ref_'+str(self.count)+'.png'))
            del depth_ref_pil
        
            with torch.no_grad():
                depth = self.s2d_update_net(torch.cat((sketch, depth_ref*(depth>0)), 1), view_code)

        depth_pil = Image.fromarray(((1-depth[0][0].detach().cpu().numpy())*255).astype(np.uint8))
        depth_pil.save(os.path.join(save_dir, 'depth_'+str(self.count)+'.png'))
        del depth_pil

        depth_pc = depth2pc(depth[0][0].detach().cpu().numpy(), azi, ele)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_pc)
        o3d.io.write_point_cloud(os.path.join(save_dir, 'depth2pc_'+str(self.count)+'.ply'), pcd)
        del depth_pc, pcd

        if edit_mask.sum() > 0:
            cv2.imwrite(os.path.join(save_dir, 'edit_mask_'+str(self.count)+'.png'), ((1-edit_mask[0])*255).astype(np.uint8))
            # render the back depth
            R_b = make_rotate(np.radians(ele), np.radians(90+azi+180), 0)
            depth_back = (self.render_depth(R_b)*0.5).unsqueeze(0)
            depth_back = torch.flip(depth_back, [3])
            depth_back[depth_back==1] = 0
            depth_back_pil = Image.fromarray(((1-depth_back[0][0].detach().cpu().numpy())*255).astype(np.uint8))
            depth_back_pil.save(os.path.join(save_dir, 'depth_back_'+str(self.count)+'.png'))
            del depth_back_pil

        # step3: iteractive volume fusion
        if self.count == 0:
            with torch.no_grad():
                self.coarse_volume = self.gas_net.generate(sketch, depth, calib)
        else:
            # zoom feature
            if self.feature_zoom_factor != self.zoom_factor:
                ratio = self.zoom_factor / self.feature_zoom_factor
                self.coarse_volume = zoom_3d(self.coarse_volume, ratio)
                self.feature_zoom_factor = self.zoom_factor
            else:
                if edit_mask.sum() > 0:
                    edit_mask = torch.from_numpy(edit_mask).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        self.coarse_volume = self.gas_net.edit_with_mask(self.coarse_volume, sketch, depth, calib, depth_ref, depth_back, edit_mask)
                else:
                    volume_tmp = self.gas_net.generate(sketch, depth, calib)
                    with torch.no_grad():
                        self.coarse_volume = self.gas_net.refine_global(self.coarse_volume, volume_tmp)
           
        # step4: surface reconstruction
        logits_list = []
        with torch.no_grad():
            self.gas_net.encoder(self.coarse_volume)
            for i in range(len(self.grid_points_split)):
                p = self.grid_points_split[i].to(self.device)
                logits = self.gas_net.decoder(p)
                del p
                logits_list.append(logits.detach().cpu())

        logits = torch.cat(logits_list, dim=1).reshape((self.output_res, self.output_res, self.output_res))

        del logits_list

        # step5: mesh generation
        self.vertices, self.triangles = mesh_from_logits(logits.numpy(), self.output_res)
        del logits

        # step6: save mesh
        self.pred_mesh = trimesh.Trimesh(self.vertices, self.triangles)
        self.pred_mesh.export(os.path.join(save_dir, 'surface_'+str(self.count)+'.obj'))

    def reset(self):
        self.vertices = None
        self.triangles = None
        self.count = -1
        self.use_normal = True
        self.coarse_volume = None
        self.zoom_factor = 1.0
        self.feature_zoom_factor = 1.0
    
    def zoom(self, zoom_factor):
        ratio = zoom_factor/self.zoom_factor
        if self.vertices is not None:
            self.vertices = self.vertices*ratio
        
        self.zoom_factor = zoom_factor
        





