import os
import trimesh
import traceback
import pickle as pkl
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp

def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    loader = dataset.get_loader(shuffle=True)
    data_tupels = []
    for i, data in tqdm(enumerate(loader)):

        export_path = out_path + '/generation/{}/'.format(data['uid'][0])
        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue
        try:
            if len(data_tupels) > 20:
                create_meshes(data_tupels)
                data_tupels = []
            logits = gen.generate_mesh(data)
            data_tupels.append((logits, data, out_path))

        except Exception as err:
            print('Error with {}: {}'.format(data['uid'][0], traceback.format_exc()))

    try:
        create_meshes(data_tupels)
        data_tupels = []
        logits = gen.generate_mesh(data)
        data_tupels.append((logits, data, out_path))

    except Exception as err:
        print('Error with {}: {}'.format(data['uid'][0], traceback.format_exc()))


def save_mesh(data_tupel):
    logits, data, out_path = data_tupel

    export_path = out_path + '/generation/{}/'.format(data['uid'][0])
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    mesh = gen.mesh_from_logits(logits[0])
    mesh.export(export_path + 'surface.off')

def create_meshes(data_tupels):
    p = mp.Pool(mp.cpu_count())
    p.map(save_mesh, data_tupels)
    p.close()
