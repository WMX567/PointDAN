import torch
import numpy as np


eps = 10e-4
eps2 = 10e-6
KL_SCALER = 10.0
MIN_POINTS = 20
RADIUS = 0.5
NROTATIONS = 4
NREGIONS=3
N = 16
K = 4
NUM_FEATURES = K * 3 + 1

def region_mean(num_regions):
    """
    Input:
        num_regions - number of regions
    Return:
        means of regions 
    """
    
    n = num_regions
    lookup = []
    d = 2 / n  # the cube size length
    #  construct all possibilities on the line [-1, 1] in the 3 axes
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            for k in range(n-1, -1, -1):
                lookup.append([1 - d * (i + 0.5), 1 - d * (j + 0.5), 1 - d * (k + 0.5)])
    lookup = np.array(lookup)
    return lookup


def assign_region_to_point(X, device, NREGIONS=3):
    """
    Input:
        X: point cloud [B, C, N]
        device: cuda:0, cpu
    Return:
        Y: Region assignment per point [B, N]
    """

    n = NREGIONS
    d = 2 / n
    X_clip = torch.clamp(X, -0.99999999, 0.99999999)  # [B, C, N]
    batch_size, _, num_points = X.shape
    Y = torch.zeros((batch_size, num_points), device=device, dtype=torch.long)  # label matrix  [B, N]

    # The code below partitions all points in the shape to voxels.
    # At each iteration find per axis the lower threshold and the upper threshold values
    # of the range according to n (e.g., if n=3, then: -1, -1/3, 1/3, 1 - there are 3 ranges)
    # and save points in the corresponding voxel if they fall in the examined range for all axis.
    region_id = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0, :]  # [B, 1, N]
                x_axis_ut = X_clip[:, 0, :] < -1 + (x + 1) * d  # [B, 1, N]
                y_axis_lt = -1 + y * d < X_clip[:, 1, :]  # [B, 1, N]
                y_axis_ut = X_clip[:, 1, :] < -1 + (y + 1) * d  # [B, 1, N]
                z_axis_lt = -1 + z * d < X_clip[:, 2, :]  # [B, 1, N]
                z_axis_ut = X_clip[:, 2, :] < -1 + (z + 1) * d  # [B, 1, N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = torch.cat([x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut,
                                      z_axis_lt, z_axis_ut], dim=1).view(batch_size, 6, -1)  # [B, 6, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask, _ = torch.min(in_range, dim=1)  # [B, N]
                Y[mask] = region_id  # label each point with the region id
                region_id += 1

    return Y

def collapse_to_point(x, device):
    """
    Input:
        X: point cloud [C, N]
        device: cuda:0, cpu
    Return:
        x: A deformed point cloud. Randomly sample a point and cluster all point
        within a radius of RADIUS around it with some Gaussian noise.
        indices: the points that were clustered around x
    """
    # get pairwise distances
    inner = -2 * torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x ** 2, dim=0, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(1, 0)

    # get mask of points in threshold
    mask = pairwise_distance.clone()
    mask[mask > RADIUS ** 2] = 100
    mask[mask <= RADIUS ** 2] = 1
    mask[mask == 100] = 0

    # Choose only from points that have more than MIN_POINTS within a RADIUS of them
    pts_pass = torch.sum(mask, dim=1)
    pts_pass[pts_pass < MIN_POINTS] = 0
    pts_pass[pts_pass >= MIN_POINTS] = 1
    indices = (pts_pass != 0).nonzero()

    # pick a point from the ones that passed the threshold
    point_ind = np.random.choice(indices.squeeze().cpu().numpy())
    point = x[:, point_ind]  # get point
    point_mask = mask[point_ind, :]  # get point mask

    # draw a gaussian centered at the point for points falling in the region
    indices = (point_mask != 0).nonzero().squeeze()
    x[:, indices] = torch.tensor(draw_from_gaussian(point.cpu().numpy(), len(indices)), dtype=torch.float).to(device)
    return x, indices


def draw_from_gaussian(mean, num_points, variance=0.001):
    """
    Input:
        mean: a numpy vector
        num_points: number of points to sample
    Return:
        points sampled around the mean with small std
    """
    return np.random.multivariate_normal(mean, np.eye(3) * variance, num_points).T


def draw_from_uniform(gap, region_mean, num_points):
    """
    Input:
        gap: a numpy vector of region x,y,z length in each direction from the mean
        region_mean:
        num_points: number of points to sample
    Return:
        points sampled uniformly in the region
    """
    return np.random.uniform(region_mean - gap, region_mean + gap, (num_points, 3)).T


def farthest_point_sample(args, xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = torch.device("cuda:" + str(xyz.get_device()) if args.cuda else "cpu")

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals

def farthest_point_sample_curv(xyz, norm_curv, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    centroids_norm_curv_vals = np.zeros((B, 4, npoint))

    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, C, 1)  # get the current chosen point value
        centroid_norm_curv = norm_curv[batch_indices, :, farthest].reshape(B, 4, 1)
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        centroids_norm_curv_vals[:, :, i] = centroid_norm_curv[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
        
    return centroids, centroids_vals, centroids_norm_curv_vals

def farthest_point_sample_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.linalg.norm(xyz - centroid, axis=1) ** 2  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    return centroids, centroids_vals


def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')


def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')


def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud.astype('float32')


def scale_to_unit_cube(x):
    """
   Input:
       x: pointcloud data, [B, C, N]
   Return:
       A point cloud scaled to unit cube
   """
    if len(x) == 0:
        return x

    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x



def get_self_supervised_label(X, k, device):
    """
    X: [B, 3, N] float tensor
    k: number of voxels per dimension (e.g. 3 → 3x3x3)
    return: transformed X1 [B, 3, N], voxel labels y [B, N]
    """
    B, C, N = X.shape
    # 1. scale to unit cube [-1, 1]
    min_coords = X.min(dim=2, keepdim=True)[0]
    max_coords = X.max(dim=2, keepdim=True)[0]
    X1 = 2 * (X - min_coords) / (max_coords - min_coords + 1e-6) - 1  # scaled to [-1, 1]

    # 2. voxelize
    d = 2 / k
    X_clipped = torch.clamp(X1, -0.999999, 0.999999)
    y = torch.zeros((B, N), device=device, dtype=torch.long)

    region_id = 0
    for xi in range(k):
        for yi in range(k):
            for zi in range(k):
                x_mask = (-1 + xi*d < X_clipped[:, 0, :]) & (X_clipped[:, 0, :] < -1 + (xi+1)*d)
                y_mask_ = (-1 + yi*d < X_clipped[:, 1, :]) & (X_clipped[:, 1, :] < -1 + (yi+1)*d)
                z_mask = (-1 + zi*d < X_clipped[:, 2, :]) & (X_clipped[:, 2, :] < -1 + (zi+1)*d)
                mask = x_mask & y_mask_ & z_mask
                y[mask] = region_id
                region_id += 1

    # 3. random permutation
    perm = torch.randperm(k**3, device=device)

    # 4. move each point to its new voxel center
    voxel_size = 2.0 / k
    centers = []
    for xi in range(k):
        for yi in range(k):
            for zi in range(k):
                cx = -1 + (xi + 0.5) * voxel_size
                cy = -1 + (yi + 0.5) * voxel_size
                cz = -1 + (zi + 0.5) * voxel_size
                centers.append([cx, cy, cz])
    centers = torch.tensor(centers, device=device).float()  # [k^3, 3]

    X_new = X1.clone()
    for b in range(B):
        for i in range(N):
            old_region = y[b, i].item()
            new_region = perm[old_region].item()
            new_center = centers[new_region]
            X_new[b, :, i] = new_center + 0.02 * torch.randn(3, device=device)  # optional augment

    return X_new, y
