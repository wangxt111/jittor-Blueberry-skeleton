import os
import argparse
import numpy as np
import trimesh
import open3d as o3d
from numpy import ndarray
from typing import Tuple, Union, Dict
from abc import ABC, abstractmethod
from collections import defaultdict

# =====================================================================================
# 1. 您提供的採樣器相關類別與函式
#    我已將它們完整地複製到這裡。
# =====================================================================================

def compute_face_weights(vertices: ndarray, faces: ndarray) -> ndarray:
    """計算用於加權採樣的面權重（基於面積的平方）。"""
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    return (face_weight ** 2).sum(axis=1)

def sample_surface_uniform(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    grid_size: int = 32,
) -> Tuple[ndarray, ndarray, ndarray]:
    """基於體素網格的表面採樣。"""
    if num_samples <= 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([]).reshape(0, 2, 1)

    face_centers = vertices[faces].mean(axis=1)
    min_bound = face_centers.min(axis=0)
    max_bound = face_centers.max(axis=0)
    
    # 防止模型為平面或一條線時出錯
    voxel_size = (max_bound - min_bound)
    voxel_size[voxel_size < 1e-6] = 1.0 
    voxel_size /= grid_size

    voxel_indices = ((face_centers - min_bound) / voxel_size).astype(int)

    voxel_dict = defaultdict(list)
    for i, idx in enumerate(map(tuple, voxel_indices)):
        voxel_dict[idx].append(i)

    all_faces_in_voxels = list(voxel_dict.values())
    total_faces_count = sum(len(f) for f in all_faces_in_voxels)
    
    if total_faces_count == 0:
         return np.array([]).reshape(0, 3), np.array([]), np.array([]).reshape(0, 2, 1)

    samples_per_voxel = [max(1, int(num_samples * len(f) / total_faces_count)) for f in all_faces_in_voxels]

    final_face_indices = []
    for face_ids, count in zip(all_faces_in_voxels, samples_per_voxel):
        face_ids = np.array(face_ids)
        weights = compute_face_weights(vertices, faces[face_ids])
        if weights.sum() < 1e-9: # 如果權重和為0，則使用均勻權重
            weights = np.ones_like(weights)
        weights /= weights.sum()
        
        # 確保不會因為浮點數精度問題而出錯
        count = min(count, len(face_ids))
        picked = np.random.choice(face_ids, size=count, p=weights, replace=True)
        final_face_indices.append(picked)

    face_index = np.concatenate(final_face_indices)
    
    # 重心座標採樣
    tri_origins = vertices[faces[face_index, 0]]
    tri_vectors = vertices[faces[face_index, 1:]] - tri_origins[:, np.newaxis, :]
    
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    mask = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[mask] = 1.0 - random_lengths[mask]

    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    
    return vertex_samples, face_index, random_lengths

class Sampler(ABC):
    """採樣器抽象基礎類別。"""
    def __init__(self):
        pass

    def _sample_barycentric(self, vertex_groups: ndarray, faces: ndarray, face_index: ndarray, random_lengths: ndarray):
        v_origins = vertex_groups[faces[face_index, 0]]
        v_vectors = vertex_groups[faces[face_index, 1:]]
        v_vectors -= v_origins[:, np.newaxis, :]
        sample_vector = (v_vectors * random_lengths).sum(axis=1)
        v_samples = sample_vector + v_origins
        return v_samples

    @abstractmethod
    def sample(self, vertices: ndarray, vertex_normals: ndarray, face_normals: ndarray, vertex_groups: dict, faces: ndarray) -> Tuple[ndarray, ndarray, dict]:
        pass

class SamplerMix(Sampler):
    """混合採樣器：一部分來自頂點，一部分來自表面。"""
    def __init__(self, num_samples: int, vertex_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.vertex_samples = vertex_samples

    def _pad_or_crop(self, x: np.ndarray, n: int):
        """將陣列的第一維度填充或裁剪到指定長度 n。"""
        if x.shape[0] == 0: # 處理空陣列
            return np.zeros((n, *x.shape[1:]), dtype=x.dtype)
        if x.shape[0] > n:
            return x[:n]
        elif x.shape[0] < n:
            # 透過重複最後一個元素來填充
            padding = np.repeat(x[-1:], n - x.shape[0], axis=0)
            return np.concatenate([x, padding], axis=0)
        return x

    def sample(self, vertices: ndarray, vertex_normals: ndarray, face_normals: ndarray, vertex_groups: dict, faces: ndarray) -> Tuple[ndarray, ndarray, dict]:
        if self.num_samples <= 0:
            return np.array([]), np.array([]), {}

        # 1. 從頂點採樣
        num_vert_samples = min(self.vertex_samples, vertices.shape[0])
        num_surf_samples = self.num_samples - num_vert_samples
        perm = np.random.permutation(vertices.shape[0])[:num_vert_samples]

        n_vertices = vertices[perm]
        n_normal = vertex_normals[perm]
        n_v = {name: v[perm] for name, v in vertex_groups.items()}

        # 2. 從表面採樣
        surf_vertices, face_idx, rand_len = sample_surface_uniform(
            num_samples=num_surf_samples,
            vertices=vertices,
            faces=faces
        )

        # 組合頂點和法線
        vertex_samples = np.concatenate([n_vertices, surf_vertices], axis=0)
        
        # 確保有法線數據可以組合
        if len(face_idx) > 0:
            surf_normals = face_normals[face_idx]
            normal_samples = np.concatenate([n_normal, surf_normals], axis=0)
        else:
            normal_samples = n_normal

        # 組合其他頂點屬性
        vertex_groups_samples = {}
        for name, v_group in vertex_groups.items():
            if len(face_idx) > 0:
                g = self._sample_barycentric(v_group, faces, face_idx, rand_len)
                vertex_groups_samples[name] = np.concatenate([n_v[name], g], axis=0)
            else:
                vertex_groups_samples[name] = n_v[name]

        # 3. 填充或裁剪以確保固定的輸出大小
        vertex_samples = self._pad_or_crop(vertex_samples, self.num_samples)
        normal_samples = self._pad_or_crop(normal_samples, self.num_samples)
        for k in vertex_groups_samples:
            vertex_groups_samples[k] = self._pad_or_crop(vertex_groups_samples[k], self.num_samples)

        return vertex_samples, normal_samples, vertex_groups_samples


# =====================================================================================
# 2. Exporter 類別 (負責儲存檔案，保持不變)
# =====================================================================================
class Exporter:
    def _safe_make_dir(self, path: str):
        dir_name = os.path.dirname(path)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
    def _export_pc(self, vertices: ndarray, path: str, vertex_normals: Union[ndarray, None] = None):
        self._safe_make_dir(path)
        if path.endswith('.ply'):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vertices)
            if vertex_normals is not None: pc.normals = o3d.utility.Vector3dVector(vertex_normals)
            o3d.io.write_point_cloud(path, pc)
        elif path.endswith('.obj'):
            with open(path, 'w') as file:
                for v in vertices: file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        else:
            raise ValueError(f"不支援的檔案擴展名: {path}。")

# =====================================================================================
# 3. 主執行函式 (已更新以使用 SamplerMix)
# =====================================================================================
def process_npz_file(input_path: str, output_path: str, num_points: int, vertex_samples_count: int):
    print("-" * 50)
    print(f"正在處理檔案: {input_path}")
    
    # 步驟 1: 載入 NPZ 檔案
    try:
        data = np.load(input_path)
        vertices = data['vertices']
        faces = data['faces']
        print(f"NPZ 檔案載入成功。包含 {len(vertices)} 個頂點和 {len(faces)} 個面。")
    except Exception as e:
        print(f"錯誤：載入 NPZ 檔案失敗: {e}")
        return

    # 步驟 2: 創建 Trimesh 物件並計算法線
    # 我們需要法線資訊來傳遞給 SamplerMix
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.vertex_normals
    mesh.face_normals
    
    # 步驟 3: 採樣
    print(f"正在將網格採樣至 {num_points} 個點 ({vertex_samples_count} 個來自頂點)...")
    # 實例化您提供的 SamplerMix
    sampler = SamplerMix(num_samples=num_points, vertex_samples=vertex_samples_count)
    # 使用 SamplerMix 的 sample 方法
    sampled_points, sampled_normals, _ = sampler.sample(
        vertices=np.asarray(mesh.vertices),
        vertex_normals=np.asarray(mesh.vertex_normals),
        face_normals=np.asarray(mesh.face_normals),
        vertex_groups={},  # 在此簡單腳本中，我們沒有額外的頂點屬性
        faces=np.asarray(mesh.faces)
    )
    print("採樣完成。")

    # 步驟 4: 儲存
    print(f"正在將採樣後的點雲儲存至: {output_path}")
    exporter = Exporter()
    try:
        exporter._export_pc(vertices=sampled_points, path=output_path, vertex_normals=sampled_normals)
        print("儲存成功！")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗: {e}")

    print("-" * 50)


# =====================================================================================
# 4. 命令列介面入口 (已更新以包含 vertex_samples 參數)
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="對 3D 網格 NPZ 檔案進行採樣，並將結果儲存為點雲。")
    
    parser.add_argument("--input", type=str, required=True, help="輸入的 .npz 檔案路徑。")
    parser.add_argument("--output", type=str, required=True, help="輸出的點雲檔案路徑 (推薦 .ply)。")
    parser.add_argument("--num_points", type=int, default=2048, help="希望採樣到的點的總數量 (預設: 2048)。")
    parser.add_argument(
        "--vertex_samples", 
        type=int, 
        default=512, 
        help="從原始頂點中直接選取的樣本數量 (預設: 512)。剩餘的點將從表面採樣。"
    )
    
    args = parser.parse_args()
    
    if args.vertex_samples > args.num_points:
        print("錯誤: --vertex_samples 的數量不能大於 --num_points。")
    else:
        process_npz_file(args.input, args.output, args.num_points, args.vertex_samples)