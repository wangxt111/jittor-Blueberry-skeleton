import numpy as np
import random
from scipy.spatial.transform import Rotation

def load_npz_data(file_path):
    """
    加载npz格式的3D人体模型数据
    Args:
        file_path: npz文件路径
    Returns:
        vertices: 顶点坐标 (N, 3)
        faces: 面片索引 (M, 3)
        joints: 骨骼节点位置 (J, 3)
        skin: 蒙皮权重 (N, J)
        parents: 骨骼层级关系 (J,)
        names: 骨骼名称 (J,)
        matrix_local: 局部变换矩阵 (J, 4, 4)
    """
    data = np.load(file_path, allow_pickle=True)
    return (data['vertices'], data['faces'], data['joints'], 
            data['skin'], data['parents'], data['names'], 
            data['matrix_local'])

def random_rotate_3d(vertices, joints, matrix_local, max_angle=30):
    """
    随机旋转3D模型
    Args:
        vertices: 顶点坐标 (N, 3)
        joints: 骨骼节点位置 (J, 3)
        matrix_local: 局部变换矩阵 (J, 4, 4)
        max_angle: 最大旋转角度（度）
    Returns:
        旋转后的顶点、骨骼和变换矩阵
    """
    # 生成随机旋转角度
    angles = np.random.uniform(-max_angle, max_angle, 3)
    # 创建旋转矩阵
    R = Rotation.from_euler('xyz', angles, degrees=True)
    
    # 旋转顶点和骨骼
    rotated_vertices = R.apply(vertices)
    rotated_joints = R.apply(joints)
    
    # 更新变换矩阵
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.as_matrix()
    rotated_matrix_local = np.array([rotation_matrix @ mat for mat in matrix_local])
    
    return rotated_vertices, rotated_joints, rotated_matrix_local

def random_scale(vertices, joints, matrix_local, scale_range=(0.9, 1.1)):
    """
    随机缩放3D模型
    Args:
        vertices: 顶点坐标 (N, 3)
        joints: 骨骼节点位置 (J, 3)
        matrix_local: 局部变换矩阵 (J, 4, 4)
        scale_range: 缩放范围
    Returns:
        缩放后的顶点、骨骼和变换矩阵
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    # 创建缩放矩阵
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    
    # 更新变换矩阵
    scaled_matrix_local = np.array([scale_matrix @ mat for mat in matrix_local])
    
    return vertices * scale, joints * scale, scaled_matrix_local

def add_gaussian_noise(vertices, noise_std=0.01):
    """
    添加高斯噪声到顶点
    Args:
        vertices: 顶点坐标 (N, 3)
        noise_std: 噪声标准差
    Returns:
        添加噪声后的顶点
    """
    noise = np.random.normal(0, noise_std, vertices.shape)
    return vertices + noise

def random_joint_perturbation(joints, max_offset=0.02):
    """
    随机扰动骨骼节点位置
    Args:
        joints: 骨骼节点位置 (J, 3)
        max_offset: 最大偏移量
    Returns:
        扰动后的骨骼节点
    """
    offset = np.random.uniform(-max_offset, max_offset, joints.shape)
    return joints + offset

def augment_3d_model(vertices, faces, joints, skin, parents, names, matrix_local,
                     rotate=True, scale=True, add_noise=True, perturb_joints=True):
    """
    综合3D模型数据增强函数
    Args:
        vertices: 顶点坐标 (N, 3)
        faces: 面片索引 (M, 3)
        joints: 骨骼节点位置 (J, 3)
        skin: 蒙皮权重 (N, J)
        parents: 骨骼层级关系 (J,)
        names: 骨骼名称 (J,)
        matrix_local: 局部变换矩阵 (J, 4, 4)
        rotate: 是否进行旋转
        scale: 是否进行缩放
        add_noise: 是否添加噪声
        perturb_joints: 是否扰动骨骼节点
    Returns:
        增强后的数据
    """
    if rotate:
        vertices, joints, matrix_local = random_rotate_3d(vertices, joints, matrix_local)
    
    if scale:
        vertices, joints, matrix_local = random_scale(vertices, joints, matrix_local)
    
    if add_noise:
        vertices = add_gaussian_noise(vertices)
    
    if perturb_joints:
        joints = random_joint_perturbation(joints)
    
    return vertices, faces, joints, skin, parents, names, matrix_local

def process_dataset(data_dir, output_dir, is_train=True):
    """
    处理整个数据集
    Args:
        data_dir: 输入数据目录
        output_dir: 输出数据目录
        is_train: 是否为训练集
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.npz'):
            # 加载数据
            file_path = os.path.join(data_dir, file_name)
            vertices, faces, joints, skin, parents, names, matrix_local = load_npz_data(file_path)
            
            # 数据增强
            if is_train:
                vertices, faces, joints, skin, parents, names, matrix_local = augment_3d_model(
                    vertices, faces, joints, skin, parents, names, matrix_local
                )
            
            # 保存增强后的数据
            output_path = os.path.join(output_dir, file_name)
            np.savez(output_path,
                    vertices=vertices,
                    faces=faces,
                    joints=joints,
                    skin=skin,
                    parents=parents,
                    names=names,
                    matrix_local=matrix_local)
