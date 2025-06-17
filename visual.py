import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_print_npz(file_path):
    # 加载.npz文件
    with np.load(file_path, allow_pickle=True) as data:
        # 打印所有键（即存储的数组名）
        print("Available keys in the .npz file:")
        for key in data.keys():
            print(key)
            # 打印对应数组的形状
            array = data[key]
            print(f"Shape of {key}: {array.shape}")

        # 获取顶点数据
        vertices = data['vertices']
        joints = data['joints']
        skin = data['skin']  # 获取蒙皮权重
        names = data['names']  # 获取骨骼名称

        # 创建一个3D图形
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制顶点
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='blue', marker='.', s=1, alpha=0.1, label='Vertices')
        
        # 绘制骨骼节点
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                  c='red', marker='o', s=50, label='Joints')

        # 在每个骨骼节点旁边标注编号和名称
        for idx, (joint, name) in enumerate(zip(joints, names)):
            ax.text(joint[0], joint[1], joint[2], f"{idx}\n{name}", 
                   color='black', fontsize=8)

        # 设置图形标题和坐标轴标签
        ax.set_title('3D Model Visualization')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        
        # 添加图例
        ax.legend()

        # 设置坐标轴比例相等
        ax.set_box_aspect([1,1,1])

        # 显示图形
        plt.show()

# 替换为你的.npz文件路径
npz_file_path = 'newdata/train/mixamo/0.npz'
print(npz_file_path)
load_and_print_npz(npz_file_path)

