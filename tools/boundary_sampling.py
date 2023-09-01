import os
import numpy as np
import trimesh
import implicit_waterproofing as iw
import multiprocessing as mp
from multiprocessing import Pool

def boundary_sampling(uid):
    off_file = os.path.join('datasets/shapenet/watertight_simplified_off', uid, 'watertight_simplified.off')
    out1_file = os.path.join('datasets/shapenet/boundary_sampling', uid, 'boundary_{}_samples.npz'.format(sigma1))
    out2_file = os.path.join('datasets/shapenet/boundary_sampling', uid, 'boundary_{}_samples.npz'.format(sigma2))

    if os.path.exists(off_file) and (not os.path.exists(out1_file) or not os.path.exists(out2_file)):
        os.makedirs(os.path.join('datasets/shapenet/boundary_sampling', uid), exist_ok=True)
        print(uid)
    
        mesh = trimesh.load(off_file)

        # 0.2
        points = mesh.sample(sample_num)
        boundary_points = points + sigma1 * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]
        grid_coords = 2 * grid_coords
        occ = iw.implicit_waterproofing(mesh, boundary_points)[0]
        np.savez(out1_file, points=boundary_points, occupancies=occ)

        # 0.015
        points = mesh.sample(sample_num)
        boundary_points = points + sigma2 * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]
        grid_coords = 2 * grid_coords
        occ = iw.implicit_waterproofing(mesh, boundary_points)[0]
        np.savez(out2_file, points=boundary_points, occupancies=occ)


if __name__ == '__main__':
    sigma1 = 0.2
    sigma2 = 0.015
    sample_num = 100000

    uids = np.loadtxt('datasets/shapenet/splits/all_02691156.lst', dtype=str)

    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, uids)
