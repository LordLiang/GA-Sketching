import numpy as np
from easydict import EasyDict as edict

cfg                                         = edict()


cfg.SHAPENET_GT_PATH = 'datasets/shapenet/watertight_simplified_off'
cfg.ANIMALHEAD_GT_PATH = 'datasets/animalhead/watertight_scaled_off'
cfg.ICONS_PATH = 'GA-Sketching-UI/sketch_3d_ui/icons/'
cfg.SHADOW_PATHS = ['GA-Sketching-UI/sketch_3d_ui/shadow/chair',
                    'GA-Sketching-UI/sketch_3d_ui/shadow/airplane',
                    'GA-Sketching-UI/sketch_3d_ui/shadow/animalhead']




