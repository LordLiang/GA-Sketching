import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.geometry import create_grid_points_from_bounds, index, orthogonal
from models.utils.img_util import dilate_3d


class GASNet_MV(nn.Module):
    def __init__(self, res, hidden_dim=256):
        super().__init__()

        self.single_view_net = GASNet(res)
        # Layer Definition
        self.merger = torch.nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 16, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 16, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

    def encoder(self, v):
        self.fea_lst = self.single_view_net.ifnet_encoder(v)
 
    def decoder(self, query):
        p = query.clone()
        p[:, :, 0], p[:, :, 2] = query[:, :, 2], query[:, :, 0]
        p = 2 * p
        p = p.unsqueeze(1).unsqueeze(1)
 
        fea_0 = F.grid_sample(self.fea_lst[0], p, mode='bilinear', padding_mode='border', align_corners=False)
        fea_1 = F.grid_sample(self.fea_lst[1], p, mode='bilinear', padding_mode='border', align_corners=False)
        fea_2 = F.grid_sample(self.fea_lst[2], p, mode='bilinear', padding_mode='border', align_corners=False)
        fea_3 = F.grid_sample(self.fea_lst[3], p, mode='bilinear', padding_mode='border', align_corners=False)
 
        del p
        features = torch.cat((fea_0, fea_1, fea_2, fea_3), dim=1)
        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))
        features = torch.cat((query.transpose(1, 2), features), dim=1)

        return self.single_view_net.decoder(features)

    def generate(self, sketch, depth, calib):
        v, _ = self.single_view_net.volume_encoder(sketch, depth, calib)
        return v

    def refine_global(self, v1, v2):
        return self.merger(torch.cat((v1, v2), 1))
        
    def edit_with_mask(self, v1, sketch, depth, calib, depth_ref, depth_back, edit_mask):
        v2, edit_mask_3d = self.single_view_net.volume_encoder(sketch, depth, calib, depth_ref, depth_back, edit_mask)
        edit_mask_3d = dilate_3d(edit_mask_3d, 5)
        v3 = v1*(1-edit_mask_3d)+v2*edit_mask_3d
        # new added!!
        v3 = self.merger(torch.cat((v3, v2), 1))
        v1 = v1*(1-edit_mask_3d)+v3*edit_mask_3d

        return v1


class GASNet(nn.Module):
    def __init__(self, res, hidden_dim=256):
        super().__init__()

        feature_size = 32 + 64 + 128 + 128 + 3
        self.volume_encoder = volume_encoder(res)
        self.ifnet_encoder = ifnet_encoder()

        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

    def decoder(self, features):
        # reconstruction branch
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        out = self.fc_out(net).squeeze(1)

        return out


class volume_encoder(nn.Module):
    def __init__(self, res, device=torch.device("cuda:0")):
        super().__init__()
        self.res = res
        self.grid_points = create_grid_points_from_bounds(-0.5, 0.5, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).float().to(device)
        self.grid_points = self.grid_points.unsqueeze(0).transpose(1, 2)

        self.conv_in_0 = nn.Conv3d(3, 16, 3, padding=1, padding_mode='replicate')
        self.conv_in_1 = nn.Conv3d(16, 16, 3, padding=1, padding_mode='replicate')
        self.conv_in_bn = nn.BatchNorm3d(16)
        self.actvn = nn.ReLU()

    def forward(self, sketch, depth, calib, depth_ref=None, depth_back=None, edit_mask=None):
        xyz = orthogonal(self.grid_points, calib) # Bx3xN
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]/2 + 0.5

        s = index(sketch, xy)
        d = index(depth, xy)
        feat = torch.cat((s, d, z), 1)
        feat = feat.reshape(1, 3, self.res, self.res, self.res)
        feat = self.actvn(self.conv_in_0(feat))
        feat = self.actvn(self.conv_in_1(feat))
        feat = self.conv_in_bn(feat)

        if edit_mask is not None:
            d_min = torch.maximum(torch.minimum(depth, depth_ref), depth_back)
            d_max = torch.maximum(depth, depth_ref)
            d_min2 = d_min[d_min>0].min()
            d_max2 = d_max[d_max>0].min()
            d_min[d_min==0] = d_min2
            d_max[d_max==0] = d_max2

            d_min = index(d_min, xy, 'nearest')
            d_max = index(d_max, xy, 'nearest')
            edit_mask = index(edit_mask, xy, 'nearest')
            replace_mask = ((z>d_min) & (z<d_max)).float()
            edit_mask = edit_mask * replace_mask
            edit_mask = edit_mask.reshape(1, 1, self.res, self.res, self.res)

        return feat, edit_mask
        

class ifnet_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')

        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)

        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        
    def forward(self, x):
        # layer 1
        x = self.actvn(self.conv_0(x))
        x = self.actvn(self.conv_0_1(x))
        x = self.conv0_1_bn(x)
        f_1 = x
        x = self.maxpool(x)

        # layer 2
        x = self.actvn(self.conv_1(x))
        x = self.actvn(self.conv_1_1(x))
        x = self.conv1_1_bn(x)
        f_2 = x
        x = self.maxpool(x)

        # layer 3
        x = self.actvn(self.conv_2(x))
        x = self.actvn(self.conv_2_1(x))
        x = self.conv2_1_bn(x)
        f_3 = x
        x = self.maxpool(x)

        # layer 4
        x = self.actvn(self.conv_3(x))
        x = self.actvn(self.conv_3_1(x))
        x = self.conv3_1_bn(x)
        f_4 = x
        return (f_1, f_2, f_3, f_4)

