import numpy as np
import open3d as o3d
import trimesh
from numpy import ndarray
from typing import Tuple, Dict, Union
from abc import ABC, abstractmethod
import os

# Exporter 和 Sampler ABC (無變更，為簡潔省略)
class Exporter:
    def _safe_make_dir(self, path: str):
        dir_name = os.path.dirname(path)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
    def _export_pc(self, vertices: ndarray, path: str, vertex_normals: Union[ndarray, None]=None, **kwargs):
        self._safe_make_dir(path)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(vertices)
        if vertex_normals is not None:
            pc.normals = o3d.utility.Vector3dVector(vertex_normals)
        o3d.io.write_point_cloud(path, pc)
        # print(f"結果已儲存至 {path}")

class Sampler(ABC):
    def __init__(self): pass
    @abstractmethod
    def sample(self, vertices: ndarray, vertex_normals: ndarray, face_normals: ndarray, vertex_groups: Dict[str, ndarray], faces: ndarray) -> Tuple[ndarray, ndarray, Dict[str, ndarray]]:
        return vertices, vertex_normals, vertex_groups

# =====================================================================================
#  【最終版】修正後的採樣器
# =====================================================================================
class SamplerEven(Sampler):
    """
    使用 trimesh.sample.sample_surface_even 進行採樣，並保證返回固定數量的點。
    此為「安靜版」，移除了所有關於採樣數量的提示性警告。
    """
    def __init__(self, num_samples: int, export_path: Union[str, None] = None):
        super().__init__()
        if num_samples <= 0: raise ValueError("num_samples must be positive.")
        self.num_samples = num_samples
        self.export_path = export_path
        if self.export_path:
            # 假設 Exporter 類別已定義
            self.exporter = Exporter()

    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: Dict[str, ndarray],
        faces: ndarray,
    ) -> Tuple[ndarray, ndarray, Dict[str, ndarray]]:
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # 初始請求數量（設為目標的4倍以提高成功率）
        n_request = self.num_samples * 4
        
        sampled_points, face_indices = trimesh.sample.sample_surface_even(mesh=mesh, count=n_request)
        
        current_num_points = len(sampled_points)

        # 處理各種情況，但不再打印任何警告
        if current_num_points == 0:
            final_points = np.repeat(mesh.vertices[0:1], self.num_samples, axis=0)
            final_face_indices = np.zeros(self.num_samples, dtype=np.int64)
        
        elif current_num_points < self.num_samples:
            # 當點數不足時，進行上採樣（但不安靜地進行）
            indices = np.arange(current_num_points)
            final_indices = np.random.choice(indices, size=self.num_samples, replace=True)
            final_points = sampled_points[final_indices]
            final_face_indices = face_indices[final_indices]
            
        else: # current_num_points >= self.num_samples
            # 當點數充足時，進行下採樣
            indices = np.arange(current_num_points)
            final_indices = np.random.choice(indices, size=self.num_samples, replace=False)
            final_points = sampled_points[final_indices]
            final_face_indices = face_indices[final_indices]

        sampled_normals = face_normals[final_face_indices]
        
        # 導出檔案的邏輯保持不變
        if self.export_path:
            self.exporter._export_pc(
                vertices=final_points,
                path=self.export_path,
                vertex_normals=sampled_normals
            )
            
        return np.asarray(final_points), np.asarray(sampled_normals), {}
