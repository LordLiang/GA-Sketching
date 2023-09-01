from evaluation import eval_mesh
import trimesh
import pickle as pkl
import os
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import random
import numpy as np
import pandas as pd


def eval(uid):
    eval_file_name = os.path.join(args.generation_path, uid, "eval.pkl")
    
    if os.path.exists(eval_file_name):
        print('File exists. Done.')
        return
    else:
        pred_mesh_path = os.path.join(args.generation_path, uid, "surface.off")
        pred_mesh = trimesh.load(pred_mesh_path, process=False)

        gt_mesh_path = os.path.join(args.gt_path, uid, "watertight_simplified.off")
        gt_mesh = trimesh.load(gt_mesh_path, process=False)

        eval = eval_mesh(pred_mesh, gt_mesh, min, max)
        pkl.dump(eval, open(eval_file_name, 'wb'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run reconstruction evaluation'
    )
    parser.add_argument('-generation_path', type=str)
    parser.add_argument('-gt_path', type=str, default='datasets/shapenet/watertight_simplified_off')
    parser.add_argument('-cat', default='airplane', type=str)
    args = parser.parse_args()

    category2id = {'airplane':'02691156', 'chair':'03001627'}
    data_list = np.loadtxt('datasets/shapenet/splits/test_{}.lst'.format(category2id[args.cat]), dtype=str)

    min = -0.5
    max = 0.5

    p = Pool(mp.cpu_count())
    # enabeling to run te script multiple times in parallel: shuffling the data
    random.shuffle(data_list)
    p.map(eval, data_list)
    p.close()
    p.join()

    # gather
    eval_all = {
        'path' : [],
        'reconst_completeness': [],
        'reconst_accuracy': [],
        'reconst_normals completeness': [],
        'reconst_normals accuracy': [],
        'reconst_normals': [],
        'reconst_completeness2': [],
        'reconst_accuracy2': [],
        'reconst_chamfer_l2': [],
        'reconst_iou' : [],
    }

    eval_all_avg = {
    }


    for uid in data_list:
        eval_path = os.path.join(args.generation_path, uid, 'eval.pkl')
        eval_reconst = pkl.load(open(eval_path,'rb'))
        eval_all['path'].append(eval_path)

        for key in eval_reconst:
            if key == 'chamfer':
                eval_all['reconst_chamfer_l2'].append( 0.5 * (eval_reconst['accuracy2'] + eval_reconst['completeness2']))
            else:
                eval_all['reconst_' + key].append(eval_reconst[key])
    pkl_dump_path = os.path.join(args.generation_path, 'evaluation_results.pkl')
    pkl.dump(eval_all, open(pkl_dump_path, 'wb'))

    for key in eval_all:
        if not key == 'path':
            data = np.array(eval_all[key])
            data = data[~np.isnan(data)]
            eval_all_avg[key+'_mean'] = np.mean(data)
            #eval_all_avg[key + '_median'] = np.median(data)
    
    print(eval_all_avg)

    avg_pkl_dump_path = os.path.join(args.generation_path, 'evaluation_results_avg.pkl')
    pkl.dump(eval_all_avg, open(avg_pkl_dump_path, 'wb'))

    eval_df = pd.DataFrame(eval_all_avg ,index=[0])
    eval_df.to_csv(os.path.join(args.generation_path, 'evaluation_results.csv'))

