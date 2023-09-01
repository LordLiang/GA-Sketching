import numpy as np
import mcubes

def make_rotate(rx, ry, rz):
    sinX, cosX = np.sin(rx), np.cos(rx)
    sinY, cosY = np.sin(ry), np.cos(ry)
    sinZ, cosZ = np.sin(rz), np.cos(rz)
    Rx = np.array([[1.0,    0,     0],
                   [  0, cosX, -sinX],
                   [  0, sinX,  cosX]])

    Ry = np.array([[ cosY,   0, sinY],
                   [    0, 1.0,    0],
                   [-sinY,   0, cosY]])

    Rz = np.array([[cosZ, -sinZ,   0],
                   [sinZ,  cosZ,   0],
                   [   0,     0, 1.0]])
    R = np.matmul(np.matmul(Rz,Ry),Rx).astype(np.float32)
    return R


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


def get_extrinsic(az, el):
    extrinsic = np.identity(4)
    R = np.matmul(make_rotate(np.radians(el), 0, 0), make_rotate(0, np.radians(-az-90), 0))
    extrinsic[0:3, 0:3] = R
    return extrinsic

intrinsic = np.array([[2,0,0,0],
                      [0,-2,0,0],
                      [0,0,2,0],
                      [0,0,0,1]])
def get_calib(az, el):
    extrinsic = get_extrinsic(az, el)
    calib = np.matmul(intrinsic, extrinsic)
    return np.array(calib, dtype=np.float32)


def mesh_from_logits(logits, res, th=0.5):
    # padding to ba able to retrieve object close to bounding box bondary
    logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
    threshold = np.log(th) - np.log(1. - th)
    vertices, triangles = mcubes.marching_cubes(logits, threshold)

    # remove translation due to padding
    vertices -= 1

    # rescale to original scale
    step = 1 / (res - 1)
    vertices = np.multiply(vertices, step)
    vertices += [-0.5, -0.5, -0.5]

    return vertices.astype(np.float32), triangles


def get_pose(az, el):
    pose = np.identity(4)
    R = make_rotate(np.radians(el), np.radians(-az-90), 0)
    pose[0:3, 0:3] = R
    return pose 


def depth2pc(depth, az, el):
    
    pose = get_pose(az, el)
    y, x = np.where(depth>0)
    depth = 0.5 - depth
    pc = np.stack([x, y, depth[y, x]], 0)
    pc[0:2,:] = pc[0:2,:] / 256 - 0.5
    pc[0:2,:] *= -1
    pc = np.dot(pose, np.concatenate([pc, np.ones((1, pc.shape[1]))], 0)).T[:, :3]
    return pc
