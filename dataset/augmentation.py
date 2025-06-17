import numpy as np
import random
from scipy.spatial.transform import Rotation
import os # 导入os模块用于文件路径操作

def load_npz_data(file_path):
    """
    加载npz格式的3D人体模型数据，并返回一个包含所有键值对的字典。
    Args:
        file_path: npz文件路径
    Returns:
        data_dict: 包含所有键值对的字典
    """
    data = np.load(file_path, allow_pickle=True)
    data_dict = {key: data[key] for key in data.keys()}
    return data_dict

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

def augment_3d_model(data_dict, rotate=True, scale=True, add_noise=False, perturb_joints=False):
    """
    综合3D模型数据增强函数
    Args:
        data_dict: 包含所有原始数据键值对的字典
        rotate: 是否进行旋转
        scale: 是否进行缩放
        add_noise: 是否添加噪声
        perturb_joints: 是否扰动骨骼节点
    Returns:
        增强后的数据字典
    """
    # 对需要增强的数据进行操作，并更新字典中的对应值
    if rotate:
        data_dict['vertices'], data_dict['joints'], data_dict['matrix_local'] = \
            random_rotate_3d(data_dict['vertices'], data_dict['joints'], data_dict['matrix_local'])
    
    if scale:
        data_dict['vertices'], data_dict['joints'], data_dict['matrix_local'] = \
            random_scale(data_dict['vertices'], data_dict['joints'], data_dict['matrix_local'])
    
    if add_noise:
        data_dict['vertices'] = add_gaussian_noise(data_dict['vertices'])
    
    if perturb_joints:
        data_dict['joints'] = random_joint_perturbation(data_dict['joints'])
    
    return data_dict # 返回整个修改后的字典

def process_dataset(data_dir, output_dir, is_train=True):
    """
    处理整个数据集
    Args:
        data_dir: 输入数据目录
        output_dir: 输出数据目录
        is_train: 是否为训练集
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.npz'):
            # 加载数据，现在load_npz_data返回一个字典
            file_path = os.path.join(data_dir, file_name)
            original_data_dict = load_npz_data(file_path)
            
            # 保存原始数据
            output_path_original = os.path.join(output_dir, file_name)
            np.savez(output_path_original, **original_data_dict)
            
            # 如果是训练集，进行数据增强并保存增强后的数据
            if is_train:
                augmented_data_dict = augment_3d_model(original_data_dict)
                # 构建增强文件的名称
                base_name, ext = os.path.splitext(file_name)
                augmented_file_name = f"{base_name}_aug{ext}"
                output_path_augmented = os.path.join(output_dir, augmented_file_name)
                np.savez(output_path_augmented, **augmented_data_dict) 
