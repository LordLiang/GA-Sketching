import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transformed_img(A, input_nc=1):
    A_transform = get_transform(grayscale=(input_nc == 1))
    return A_transform(A)


def dilate(bin_img, ksize=3):
    pad = (ksize - 1) // 2
    bin_img_pad = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img_pad, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=3):
    out = 1 - dilate(1-bin_img, ksize)
    return out


def open(bin_img, ksize=3):
    out = erode(bin_img, ksize)
    out = dilate(out, ksize)
    return out


def close(bin_img, ksize=3):
    out = dilate(bin_img, ksize)
    out = erode(out, ksize)
    return out


def dilate_3d(bin_vox, ksize=3):
    pad = (ksize - 1) // 2
    bin_vox_pad = F.pad(bin_vox, pad=[pad, pad, pad, pad, pad, pad], mode='reflect')
    out = F.max_pool3d(bin_vox_pad, kernel_size=ksize, stride=1, padding=0)
    return out

def erode_3d(bin_vox, ksize=3):
    out = 1 - dilate_3d(1-bin_vox, ksize)
    return out


def zoom_3d(tensor, ratio):
    pad = int(32*np.abs(1-ratio))
    print(ratio, pad)
    if ratio < 1:
        size = (64-2*pad, 64-2*pad, 64-2*pad)
        tensor = F.interpolate(tensor, size, mode='trilinear', align_corners=False)
        tensor = F.pad(tensor, pad=[pad, pad, pad, pad, pad, pad], mode='replicate')
    elif ratio > 1:
        size = (64+2*pad, 64+2*pad, 64+2*pad)
        tensor = F.interpolate(tensor, size, mode='trilinear', align_corners=False)
        tensor = tensor[:,:,pad:(64+pad), pad:(64+pad), pad:(64+pad)]
    return tensor
    




