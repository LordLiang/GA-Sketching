import torch
import torch.nn as nn
import torch.nn.functional as F
from core.geometry import create_grid_points_from_bounds, index, orthogonal

class GASNet_MV(nn.Module):
    def __init__(self, res, category, hidden_dim=256):
        super().__init__()

        self.single_view_net = GASNet(res)

        if category == 'chair':
            checkpoint = 'pretrained/single_view/chair/checkpoint_epoch_200.tar'
        else:
            checkpoint = 'pretrained/single_view/airplane/checkpoint_epoch_200.tar'
        self.single_view_net.load_state_dict(torch.load(checkpoint))
        self.freeze_module(self.single_view_net)

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

    def freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def encoder(self, img_lst, calib_lst, num_views):
        v = self.single_view_net.volume_encoder(img_lst[0], calib_lst[0])
        # iterative aggregation
        for i in range(1, num_views):
            m = self.single_view_net.volume_encoder(img_lst[i], calib_lst[i])
            v = self.merger(torch.cat((v, m), 1))
        return self.single_view_net.ifnet_encoder(v)

    def decoder(self, query, fea_lst):
        out = self.single_view_net.decoder(query, fea_lst)
        return out

    def forward(self, query, img_lst, calib_lst, num_views):
        fea_lst = self.encoder(img_lst, calib_lst, num_views)
        logits = self.decoder(query, fea_lst)
        return logits

    def forward_with_split(self, query_split, img_lst, calib_lst, num_views, device):
        fea_lst = self.encoder(img_lst, calib_lst, num_views)
        logits_lst = []
        for i in range(len(query_split)):
            query = query_split[i].to(device)
            logits = self.decoder(query, fea_lst)
            logits_lst.append(logits.detach().cpu())

        logits = torch.cat(logits_lst, dim=1).numpy()

        return logits
 

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

    def encoder(self, img, calib):
        v = self.volume_encoder(img, calib)
        return self.ifnet_encoder(v)

    def decoder(self, query, fea_lst):
        p = query.clone()
        p[:, :, 0], p[:, :, 2] = query[:, :, 2], query[:, :, 0]
        p = 2 * p
        p = p.unsqueeze(1).unsqueeze(1)
 
        fea_0 = F.grid_sample(fea_lst[0], p, mode='bilinear', padding_mode='border', align_corners=False)
        fea_1 = F.grid_sample(fea_lst[1], p, mode='bilinear', padding_mode='border', align_corners=False)
        fea_2 = F.grid_sample(fea_lst[2], p, mode='bilinear', padding_mode='border', align_corners=False)
        fea_3 = F.grid_sample(fea_lst[3], p, mode='bilinear', padding_mode='border', align_corners=False)
 
        del p
        features = torch.cat((fea_0, fea_1, fea_2, fea_3), dim=1)
        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))
        features = torch.cat((query.transpose(1, 2), features), dim=1)

        # reconstruction branch
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        out = self.fc_out(net).squeeze(1)
        return out

    def forward(self, query, img, calib):
        fea_lst = self.encoder(img, calib)
        logits = self.decoder(query, fea_lst)
        return logits

    def forward_with_split(self, query_split, img, calib, device):
        fea_lst = self.encoder(img, calib)
        logits_lst = []
        for i in range(len(query_split)):
            query = query_split[i].to(device)
            logits = self.decoder(query, fea_lst)
            logits_lst.append(logits.detach().cpu())

        logits = torch.cat(logits_lst, dim=1).numpy()

        return logits



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

    def forward(self, img, calib):
        B = img.shape[0]
        p = self.grid_points.repeat(B, 1, 1)
        xyz = orthogonal(p, calib) # Bx3xN
        xy = xyz[:, :2, :]

        img_feat = index(img, xy)
        z_feat = xyz[:, 2:3, :]/2 + 0.5
        
        feat = torch.cat((img_feat, z_feat), 1)
        feat = feat.reshape(B, 3, self.res, self.res, self.res)
        feat = self.actvn(self.conv_in_0(feat))
        feat = self.actvn(self.conv_in_1(feat))
        feat = self.conv_in_bn(feat)

        return feat


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

