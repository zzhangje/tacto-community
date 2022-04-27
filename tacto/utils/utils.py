import numpy as np
from scipy.spatial.transform import Rotation as R


def gen_quat_t(quat):
    '''convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrix '''
    T = np.zeros((4, 4))
    T[:3,:3] = R.from_quat(quat[3:]).as_matrix()
    T[0:3,3] = quat[:3] # t
    T[3, 3] = 1
    return T

def gen_t_quat(pose):
    '''convert 4 x 4 transformation matrix to [x, y, z, qx, qy, qz, qw]'''
    r = R.from_matrix(pose[0:3,0:3])
    q = r.as_quat() # qx, qy, qz, qw
    t = pose[0:3,3].T
    return np.concatenate((t, q), axis=0)

def skewMat(v):
    v = np.atleast_2d(v)
    # vector to its skew matrix
    mat = np.zeros((3,3, v.shape[0]))
    mat[0,1, :] = -1*v[:, 2]
    mat[0,2, :] = v[:, 1]

    mat[1, 0, :] = v[:, 2]
    mat[1, 2, :] = -1*v[:, 0]

    mat[2, 0, :] = -1*v[:, 1]
    mat[2, 1, :] = v[:, 0]
    return mat

def normalize(v):
    # normalize the vector
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def padding(img):
    # pad one row & one col on each side
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), 'edge')
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'edge')

def gen_pose(vertices, normals, shear_mag, delta):
    vertices = np.atleast_2d(vertices)
    normals = np.atleast_2d(normals)
    
    num_samples = vertices.shape[0]
    T = np.zeros((4, 4, num_samples)) # transform from point coord to world coord
    T[3, 3, :] = 1
    T[0:3,3, :] = vertices.T # t

    # resolve ambiguous DoF 
    '''Find rotation of shear_vector so its orientation matches normal: np.dot(Rot, shear_vector) = normal
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another '''
    
    cos_shear_mag = np.random.uniform(low=np.cos(shear_mag), high=1.0, size=(num_samples,))        # Base of shear cone
    shear_phi = np.random.uniform(low=0.0, high=2*np.pi, size=(num_samples,)) # Circle of shear cone
    # TODO: different shear vectors 
    # v = np.zeros((num_samples, 3))
    # for i in range(num_samples):
    #     shear_vector = np.array([np.sqrt(1-cos_shear_mag[i]**2)*np.cos(shear_phi[i]), np.sqrt(1-cos_shear_mag[i]**2)*np.sin(shear_phi[i]), cos_shear_mag[i]])
    #     shear_vector = shear_vector.reshape(1, -1)
    #     v[i, :] = np.cross(shear_vector, normals[i, :])     

    # Axis v = (shear_vector \cross normal)/(||shear_vector \cross normal||)
    shear_vector = np.array([np.sqrt(1-cos_shear_mag**2)*np.cos(shear_phi), np.sqrt(1-cos_shear_mag**2)*np.sin(shear_phi), cos_shear_mag]).T
    shear_vector_skew = skewMat(shear_vector)
    v = np.einsum('ijk,jk->ik', shear_vector_skew, normals.T).T
    v = v/np.linalg.norm(v, axis=1).reshape(-1, 1) # ITS THE FIX!

    # find corner cases 
    zero_idx = np.isclose(np.linalg.norm(v, axis=1), 0.0, atol=0.1) #handle https://math.stackexchange.com/a/293130
    # zero_idx_up = zero_idx
    # zero_idx_down = False
    zero_idx_up = np.logical_and(zero_idx,normals[:, 2] > 0) # pointing up 
    zero_idx_down = np.logical_and(zero_idx,normals[:, 2] < 0) # pointing down

    v_skew, sampledNormals_skew = skewMat(v), skewMat(normals)

    # Angle theta = \arccos(z_axis \dot normal) 
    # elementwise: theta = np.arccos(np.dot(shear_vector,normal)/(np.linalg.norm(shear_vector)*np.linalg.norm(normal)))
    theta = np.arccos(np.einsum('ij,ij->i', shear_vector, normals)/(np.linalg.norm(shear_vector, axis = 1)*np.linalg.norm(normals, axis = 1)))

    identity_3d = np.zeros(v_skew.shape)
    np.einsum('iij->ij', identity_3d)[:] = 1
    # elementwise: Rot = np.identity(3) + v_skew*np.sin(theta) + np.linalg.matrix_power(v_skew,2) * (1-np.cos(theta)) # rodrigues
    Rot = identity_3d + v_skew*np.sin(theta) + np.einsum('ijn,jkn->ikn', v_skew, v_skew) * (1-np.cos(theta)) # rodrigues

    # for i in range(Rot.shape[2]):
    #     if np.linalg.det(Rot[:, :, i]) == 0:
    #         Rot[:, :, i] =  np.identity(3)
    
    if np.any(zero_idx_up):
        Rot[:3,:3, zero_idx_up] = np.dstack([np.identity(3)]*np.sum(zero_idx_up))
    if np.any(zero_idx_down):
        Rot[:3,:3, zero_idx_down] = np.dstack([np.array([[1,  0,  0], [0, -1, -0], [0,  0, -1]])]*np.sum(zero_idx_down))

    # Rotation about Z axis is still ambiguous, generating random rotation b/w [0, 2pi] about normal axis
    # elementwise: RotDelta = np.identity(3) + normal_skew*np.sin(delta[i]) + np.linalg.matrix_power(normal_skew,2) * (1-np.cos(delta[i])) # rodrigues
    RotDelta = identity_3d + sampledNormals_skew*np.sin(delta) + np.einsum('ijn,jkn->ikn', sampledNormals_skew, sampledNormals_skew) * (1-np.cos(delta)) # rodrigues

    # elementwise:  RotDelta @ Rot
    T[:3,:3, :] = np.einsum('ijn,jkn->ikn', RotDelta, Rot)
    return T
