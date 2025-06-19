import numpy as np
import random
from scipy.spatial.transform import Rotation
import os
from typing import Dict, Tuple

# ==============================================================================
# 1. 改進後的物件導向增強器 (Augmenter Class)
# ==============================================================================
class ModelAugmenter:
    """
    一個物件導向的3D人體模型數據增強器。
    
    該類將多種數據增強方法封裝在一起，提供統一的配置和調用介面。
    借鑒了PointNet等點雲處理框架中的豐富增強策略。
    """
    def __init__(self,
                 mirror_p: float = 0.5,
                 scale_range: Tuple[float, float] = (0.85, 1.15),
                 rotation_max_angles: Tuple[float, float, float] = (10, 30, 10),
                 shift_range: float = 0.05,
                 vertex_noise_std: float = 0.001,
                 joint_noise_std: float = 0.002):
        """
        初始化增強器並配置所有參數。

        Args:
            mirror_p (float): 執行X軸鏡像的機率。對於人體模型，通常只鏡像X軸。
            scale_range (tuple): 均勻縮放的範圍 (min, max)。
            rotation_max_angles (tuple): (x, y, z)軸的最大隨機旋轉角度(度)。
            shift_range (float): 沿各軸隨機平移的最大範圍（單位：米）。
            vertex_noise_std (float): 添加到頂點的高斯噪聲（抖動）的標準差。
            joint_noise_std (float): 添加到關節點的高斯噪聲（擾動）的標準差。
        """
        self.mirror_p = mirror_p
        self.scale_range = scale_range
        self.rotation_max_angles = rotation_max_angles
        self.shift_range = shift_range
        self.vertex_noise_std = vertex_noise_std
        self.joint_noise_std = joint_noise_std
        print("ModelAugmenter 已初始化，將按以下順序執行增強：鏡像 -> 縮放 -> 旋轉 -> 平移 -> 噪聲")

    def _apply_transform(self, vertices, joints, matrix_local, T):
        """通用輔助函式，應用一個4x4的變換矩陣T。"""
        # 變換頂點和關節點
        # (N, 3) @ (3, 3).T -> (N, 3), 加上平移部分
        points_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        transformed_vertices = (T @ points_homo.T).T[:, :3]

        joints_homo = np.hstack([joints, np.ones((joints.shape[0], 1))])
        transformed_joints = (T @ joints_homo.T).T[:, :3]
        
        # 更新局部變換矩陣: M' = T @ M
        # 使用向量化操作 (T @ matrix_local) 替代 list comprehension，效率更高
        transformed_matrix_local = T @ matrix_local
        
        return transformed_vertices, transformed_joints, transformed_matrix_local

    def _random_mirror(self, vertices, joints, matrix_local):
        """沿X軸進行隨機鏡像 (最適合人體模型)。"""
        if random.random() < self.mirror_p:
            S = np.diag([-1, 1, 1, 1]) # X軸鏡像矩陣
            return self._apply_transform(vertices, joints, matrix_local, S)
        return vertices, joints, matrix_local

    def _random_scale(self, vertices, joints, matrix_local):
        """隨機均勻縮放。"""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        S = np.diag([scale, scale, scale, 1])
        # 注意：此處的實現修正了原程式碼的問題。
        # 正確做法是用縮放矩陣去左乘變換矩陣，而不是直接操作矩陣的平移部分。
        return self._apply_transform(vertices, joints, matrix_local, S)

    def _random_rotate(self, vertices, joints, matrix_local):
        """隨機旋轉。"""
        angles = [np.random.uniform(-a, a) for a in self.rotation_max_angles]
        R_mat = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        return self._apply_transform(vertices, joints, matrix_local, T)
        
    def _random_shift(self, vertices, joints, matrix_local):
        """借鑒PointNet：隨機平移整個模型。"""
        shift_vec = np.random.uniform(-self.shift_range, self.shift_range, 3)
        T = np.eye(4)
        T[:3, 3] = shift_vec
        return self._apply_transform(vertices, joints, matrix_local, T)

    def _add_noise(self, vertices, joints):
        """借鑒PointNet：添加抖動噪聲和關節點擾動。"""
        if self.vertex_noise_std > 0:
            vertices += np.random.normal(0, self.vertex_noise_std, vertices.shape)
        if self.joint_noise_std > 0:
            joints += np.random.normal(0, self.joint_noise_std, joints.shape)
        return vertices, joints

    def augment(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        執行完整的數據增強流程。

        Args:
            data_dict (dict): 包含 'vertices', 'joints', 'matrix_local' 的原始數據字典。

        Returns:
            dict: 增強後的數據字典。
        """
        # 深度複製以避免修改原始數據
        aug_data = {k: v.copy() for k, v in data_dict.items() if isinstance(v, np.ndarray)}
        
        v, j, m = aug_data['vertices'], aug_data['joints'], aug_data['matrix_local']

        # 依次應用增強
        v, j, m = self._random_mirror(v, j, m)
        v, j, m = self._random_scale(v, j, m)
        v, j, m = self._random_rotate(v, j, m)
        v, j, m = self._random_shift(v, j, m)
        v, j = self._add_noise(v, j)
        
        aug_data['vertices'], aug_data['joints'], aug_data['matrix_local'] = v, j, m
        return aug_data

# ==============================================================================
# 2. 數據集處理函式 (現在變得更簡潔)
# ==============================================================================

def load_npz_data(file_path: str) -> Dict[str, np.ndarray]:
    """加載npz文件並返回字典。"""
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}

def process_dataset(data_dir: str, output_dir: str, augmenter: ModelAugmenter, is_train: bool = True):
    """
    處理整個數據集。
    
    Args:
        data_dir (str): 輸入數據目錄。
        output_dir (str): 輸出數據目錄。
        augmenter (ModelAugmenter): 用於數據增強的增強器實例。
        is_train (bool): 是否為訓練集（決定是否執行增強）。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n開始處理目錄: {data_dir}")
    
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not file_list:
        print("錯誤：在輸入目錄中沒有找到 .npz 文件。")
        return
        
    for i, file_name in enumerate(file_list):
        print(f"  處理中 ({i+1}/{len(file_list)}): {file_name}")
        file_path = os.path.join(data_dir, file_name)
        
        original_data_dict = load_npz_data(file_path)
        
        # 保存一份原始數據
        output_path_original = os.path.join(output_dir, file_name)
        np.savez(output_path_original, **original_data_dict)
        
        if is_train:
            # 使用增強器來獲取增強後的數據
            augmented_data_dict = augmenter.augment(original_data_dict)
            
            base_name, ext = os.path.splitext(file_name)
            augmented_file_name = f"{base_name}_aug{ext}"
            output_path_augmented = os.path.join(output_dir, augmented_file_name)
            np.savez(output_path_augmented, **augmented_data_dict)
            
    print(f"處理完成！數據已保存至: {output_dir}")

# --- 步驟 1: 創建一個增強器實例 ---
# 你可以在這裡輕鬆配置所有參數
augmenter = ModelAugmenter(
    mirror_p=0.5,
    scale_range=(0.9, 1.1),
    rotation_max_angles=(5, 20, 5), # Y軸旋轉範圍更大
    shift_range=0.03,
    vertex_noise_std=0.0005,
    joint_noise_std=0.001
)

# --- 步驟 3: 執行數據處理流程 ---
process_dataset(
    data_dir='data/train/mixamo', 
    output_dir='newdata/train/mixamo', 
    augmenter=augmenter, 
    is_train=True
)

process_dataset(
    data_dir='data/train/vroid', 
    output_dir='newdata/train/vroid', 
    augmenter=augmenter, 
    is_train=True
)

def update_train_list(train_list_path):
    """
    读取train_list.txt，为每个训练数据文件添加_aug后缀的新文件名。
    """
    with open(train_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if line.startswith('train/vroid/') or line.startswith('train/mixamo/'):
            base_name, ext = os.path.splitext(line)
            new_lines.append(f"{base_name}_aug{ext}")

    # 将更改写入新的文件
    output_train_list_path = os.path.join(os.path.dirname(train_list_path), 'train_list_1.txt')
    with open(output_train_list_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')
    print(f"Updated {output_train_list_path} with augmented file names.")

# 调用函数更新 train_list.txt
update_train_list('data/train_list.txt')

print('done.')