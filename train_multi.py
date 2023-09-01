import argparse
import torch
from core.models.gas_model import GASNet_MV
import core.data.voxelized_data_shapenet as voxelized_data
from core.trainer import Trainer

# common option
parser = argparse.ArgumentParser(description='Run Model')
parser.add_argument('-dist','--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
parser.add_argument('-std_dev','--sample_sigmas',default=[0.2, 0.015], nargs='+', type=float)
parser.add_argument('-batch_size' , default=4, type=int)
parser.add_argument('-epoch' , default=200, type=int)
parser.add_argument('-o','--optimizer' , default='Adam', type=str)
parser.add_argument('-num_sample_points', default=10000, type=int)
parser.add_argument('-cat','--category' , default='airplane', type=str)#'chair' or 'airplane'
# sgnet option
parser.add_argument('-res' , default=64, type=int)

args = parser.parse_args()

net = GASNet_MV(args.res, args.category)

exp_name = '{}v{}_{}'.format('GAS_MV', args.res, args.category)


train_dataset = voxelized_data.VoxelizedDataset('train', args.category,
                                                sample_distribution=args.sample_distribution,
                                                sample_sigmas=args.sample_sigmas,
                                                num_sample_points=args.num_sample_points, 
                                                batch_size=args.batch_size, 
                                                num_workers=8)

val_dataset = voxelized_data.VoxelizedDataset('val', args.category,
                                              sample_distribution=args.sample_distribution,
                                              sample_sigmas=args.sample_sigmas,
                                              num_sample_points=args.num_sample_points, 
                                              batch_size=args.batch_size, 
                                              num_workers=8)

trainer = Trainer(net,
                  torch.device("cuda"), 
                  train_dataset, 
                  val_dataset, 
                  exp_name, 
                  optimizer=args.optimizer,
                  stage='multi')

trainer.train_model(args.epoch+1, args.num_sample_points)
