import argparse
from core.models.gas_model import GASNet
import core.data.voxelized_data_shapenet as voxelized_data
from core.generator import Generator
from core.generation_iterator import gen_iterator

parser = argparse.ArgumentParser(description='Run generation')
parser.add_argument('-res' , default=64, type=int)
parser.add_argument('-retrieval_res' , default=128, type=int)
parser.add_argument('-checkpoint', type=int)
parser.add_argument('-cat','--category' , default='airplane', type=str)#'chair' or 'airplane'
args = parser.parse_args()

net = GASNet(args.res)

exp_name = '{}v{}_{}'.format('GAS', args.res, args.category)

dataset = voxelized_data.VoxelizedDataset('test', 
                                          args.category,
                                          batch_size=1, 
                                          num_workers=0)

gen = Generator(net,
                0.5,
                exp_name, 
                checkpoint=args.checkpoint,
                resolution=args.retrieval_res,
                num_views=1)

out_path = 'experiments/{}/evaluation_{}@{}_{}v/'.format(exp_name, args.checkpoint, args.retrieval_res, 1)
gen_iterator(out_path, dataset, gen)

