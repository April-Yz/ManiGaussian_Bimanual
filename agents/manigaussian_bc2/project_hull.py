import numpy as np
import cv2
from scipy.spatial import ConvexHull, Delaunay, QhullError
import torch
import time

def project_points_3d_to_2d(points_3d, intrinsic_matrix):
    """
    :param points_3d: (N, 3) numpy 
    :param intrinsic_matrix: 3x3 
    :return: (N, 2) numpy 
    """
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)
    
    return np.stack((u, v), axis=-1)

def mark_points_in_mask(points_3d, mask, intrinsic_matrix, depth_map):
    """
    no use
    :param points_3d: (N, 3) numpy
    :param mask: (H, W) numpy
    :param intrinsic_matrix: 3x3
    :param depth_map: (H, W) numpy 
    :return: 
    """
    # 3D -> 2D
    points_2d = project_points_3d_to_2d(points_3d, intrinsic_matrix)
    
    # in mask?
    for u, v in points_2d:
        if u >= 0 and u < mask.shape[1] and v >= 0 and v < mask.shape[0]:
            if mask[v, u] > 0:
                cv2.circle(mask, (u, v), 3, (255, 0, 0), -1)  
    return mask


def label_point_cloud(points, D, K, mask): 
    """
    2D->3D
    points : np.ndarray  
    D : np.ndarray       
    K : np.ndarray       
    mask : np.ndarray    
    labeled_points : np.ndarray   
    """
    fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item()
    cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()
    labeled_points = []

    for X, Y, Z in points:
        u = int((fx * X / Z) + cx)
        v = int((fy * Y / Z) + cy)
        if 0 <= u < D.shape[1] and 0 <= v < D.shape[0]:
            depth = D[v, u]
            if depth > 0 and mask[v, u] > 0: 
                label = mask[v, u]
                labeled_points.append((X, Y, Z, label))

    return np.array(labeled_points)  


def project_3d_to_2d_CPU(points, K): # 1.5s
    """
    points : np.ndarray 
    K : np.ndarray       
    projected_points : np.ndarray   
    """
    projected_points = []
    for X, Y, Z, B in points:
        if B:
            u = (K[0, 0, 0] * X / Z) + K[0, 0, 2]  # x
            v = (K[0, 1, 1] * Y / Z) + K[0, 1, 2]  # y 
            projected_points.append((u, v))
    
    return np.array(projected_points)

def project_3d_to_2d(points, K):
    """
    points : torch.Tensor 
    K : torch.Tensor      
    projected_points : torch.Tensor   
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    B = points[:, 3].to(torch.bool)

    u = (K[0, 0, 0] * X / Z) + K[0, 0, 2]
    v = (K[0, 1, 1] * Y / Z) + K[0, 1, 2]

    projected_points = torch.stack((u[B], v[B]), dim=1)

    return projected_points



def depth_mask_to_3d(D, mask, K): 
    """
    D : np.ndarray      
    mask : np.ndarray   
    K : np.ndarray      
    labeled_points : np.ndarray 
    """
    fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item() 
    cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item() 

    labeled_points = []

    D = D[0][0]                     #[256,256]
    mask = mask[0][0]

    valid_mask = (mask >= 94) & (mask <= 114)
    y_idxs, x_idxs = torch.where(valid_mask)
    depths = D[y_idxs, x_idxs]
    X_cam = (x_idxs.float() - cx) * depths / fx
    Y_cam = (y_idxs.float() - cy) * depths / fy
    Z_cam = depths
    labeled_points = torch.stack((X_cam, Y_cam, Z_cam), dim=-1)
    return labeled_points  


def points_inside_convex_hull(point_cloud, masked_points, remove_outliers=False, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.

    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud. 
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull. 
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull
                                            and False otherwise.
    """

    # Remove outliers if the option is selected 
    if remove_outliers:                                                 # Default is True
        Q1 = np.percentile(masked_points, 0, axis=0)
        Q3 = np.percentile(masked_points, 80, axis=0)                  
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR)) 
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    if filtered_masked_points.shape[0] < 4:
        return torch.cat([point_cloud, torch.zeros((point_cloud.shape[0], 1), device=point_cloud.device)], dim=1)
    try:
        delaunay = Delaunay(filtered_masked_points)
    except QhullError as e:
        print(f"Found Error: {e}")
        return torch.cat([point_cloud, torch.zeros((point_cloud.shape[0], 1), device=point_cloud.device)], dim=1)
    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0
    count_inside_hull = np.sum(points_inside_hull_mask)
    print("count_inside_hull = ", count_inside_hull)
    points_inside_hull_mask = torch.cat([point_cloud, torch.tensor(points_inside_hull_mask, device=point_cloud.device).unsqueeze(1)], dim=1)
    return points_inside_hull_mask


def create_2d_mask_from_convex_hull_CPU(points_2d, shape):
    """
    points_2d : np.ndarray
    shape : tuple
    mask : np.ndarray
    """
    # mask = np.zeros(shape, dtype=np.uint8)
    mask = torch.ones(shape, dtype=torch.uint8)
    if points_2d.shape[0] ==0:
        return mask
    else:
        delaunay = Delaunay(points_2d)
        
        for x in range(shape[1]):
            for y in range(shape[0]):
                if delaunay.find_simplex((x, y)) >= 0:
                    mask[y, x] = 0  
        
        return mask

def get_convex_hull_mask_CPU(points_2d, image_shape):
    """
    Computes the convex hull of the 2D points and creates a mask.
    """
    hull = cv2.convexHull(points_2d.astype(np.int32))

    mask = np.zeros(image_shape, dtype=np.uint8)     
    cv2.fillConvexPoly(mask, hull, (255,255,255),lineType=cv2.LINE_AA)
    return mask


def create_2d_mask_from_convex_hull(points_2d, shape):
    """
        points_2d : np.ndarray 
        shape : tuple 
        mask : torch.Tensor
    """
    mask = torch.ones(shape, dtype=torch.uint8)
    
    if (points_2d.shape[0] < 3):
        return mask
    try:
        points_2d = points_2d.cpu().numpy()
        delaunay = Delaunay(points_2d)
    except QhullError as e:
        print(f"Error: {e}")
        return mask
    
    grid_x, grid_y = torch.meshgrid(torch.arange(shape[1]), torch.arange(shape[0]), indexing='ij')

    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).numpy()

    simplex_indices = delaunay.find_simplex(grid_points)

    mask = mask.flatten()
    mask[simplex_indices >= 0] = 0
    mask = mask.view(shape)
    
    return mask

def merge_arrays(array1, array2):
    if array2.size == 0: 
        return array1  
    
    if array1.size == 0:
        return array2  
    merged_array = np.concatenate((array1, array2), axis=0)
    return merged_array

def merge_tensors(tensor1, tensor2):
    if tensor2.size(0) == 0:  
        return tensor1  
    
    if tensor1.size(0) == 0:
        return tensor2  
    
    merged_tensor = torch.cat((tensor1, tensor2), dim=0)
    return merged_tensor



def test_points_inside_convex_hull():
    point_cloud = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0]
    ], dtype=torch.float32)

    masked_points = point_cloud[:4]

    inside_hull_mask = points_inside_convex_hull(point_cloud, masked_points, remove_outliers=False)

    print("Points inside convex hull mask:")
    print(inside_hull_mask)

def test_create_2d_mask_from_convex_hull():
    points_2d = torch.tensor([
        [2, 2],
        [4, 4],
        [8, 8],
        [10, 10]
    ], dtype=torch.float32)

    mask_shape = (10, 10)
    masked_points = points_2d[:2]
    print("masked_points",masked_points,masked_points.shape)
    mask = create_2d_mask_from_convex_hull(masked_points, mask_shape)

    print(mask.numpy())


def main():
    test_points_inside_convex_hull()
    print("test successful")

if __name__ == "__main__":
    main()
