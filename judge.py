import jittor as jt
import numpy as np
import argparse
import os
import shutil # 用于清理 __pycache__

# 定义 J2J 函数 (与之前相同)
def J2J(
    joints_a: jt.Var,
    joints_b: jt.Var,
) -> jt.Var:
    '''
    计算在 [-1, 1]^3 立方体空间中的 J2J 损失。
    
    joints_a: (J1, 3) 关节坐标
    joints_b: (J2, 3) 关节坐标
    '''
    assert isinstance(joints_a, jt.Var)
    assert isinstance(joints_b, jt.Var)
    assert joints_a.ndim == 2, "joints_a 的形状应为 (J1, 3)"
    assert joints_b.ndim == 2, "joints_b 的形状应为 (J2, 3)"
    
    dis1 = ((joints_a.unsqueeze(0) - joints_b.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss1 = dis1.min(dim=-1)[0]
    
    dis2 = ((joints_b.unsqueeze(0) - joints_a.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss2 = dis2.min(dim=-1)[0]
    
    # 你的 J2J 函数在末尾有两个 / 2，我保留了原始逻辑。
    # 通常 J2J 是 (loss1.mean() + loss2.mean()) / 2
    # 如果意图是进一步归一化，则 / 2 / 2 是可以的
    return (loss1.mean() + loss2.mean()) / 2 / 2 

def get_joints_from_npz(file_path: str, key_name: str) -> np.ndarray:
    """
    从 NPZ 文件中加载关节数据，智能处理单数组或多数组情况。
    """
    try:
        loaded_data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"文件未找到: '{file_path}'. 请检查路径是否正确。")
    except Exception as e:
        raise Exception(f"加载 NPZ 文件 '{file_path}' 时发生错误: {e}")

    # 判断加载的数据是NpzFile对象（多数组）还是直接的ndarray（单数组）
    if isinstance(loaded_data, np.lib.npyio.NpzFile):
        # 如果是多数组，尝试按key_name获取
        if key_name not in loaded_data:
            raise KeyError(
                f"错误: 文件 '{file_path}' 中未找到键 '{key_name}'。\n"
                f"'{file_path}' 中可用的键: {list(loaded_data.keys())}"
            )
        joints_np = loaded_data[key_name]
    elif isinstance(loaded_data, np.ndarray):
        # 如果是单数组，直接使用
        print(f"注意: 文件 '{file_path}' 直接包含一个 NumPy 数组，忽略键 '{key_name}'。")
        joints_np = loaded_data
    else:
        raise TypeError(f"文件 '{file_path}' 加载了未知类型的数据: {type(loaded_data)}")

    # 验证关节数据形状
    if joints_np.ndim != 2 or joints_np.shape[-1] != 3:
        raise ValueError(
            f"文件 '{file_path}' 中关节数据形状为 {joints_np.shape}，预期为 (N, 3)。"
        )
    return joints_np

def calculate_j2j_from_npz_files(
    file_path_a: str, 
    file_path_b: str, 
    key_name_a: str = 'joints', 
    key_name_b: str = 'joints'
) -> float:
    """
    从两个 NPZ 文件中加载关节数据并计算 J2J 指标。

    Args:
        file_path_a (str): 第一个 NPZ 文件的路径。
        file_path_b (str): 第二个 NPZ 文件的路径。
        key_name_a (str): 第一个 NPZ 文件中存储关节数据的键名，默认为 'joints'。
        key_name_b (str): 第二个 NPZ 文件中存储关节数据的键名，默认为 'joints'。

    Returns:
        float: 计算得到的 J2J 指标。
    
    Raises:
        FileNotFoundError: 如果未找到任一文件。
        KeyError: 如果在多数组文件中未找到指定的键。
        ValueError: 如果加载的关节数据形状不正确。
        TypeError: 如果加载了未知类型的数据。
    """
    
    # 使用新的辅助函数加载关节数据
    joints_a_np = get_joints_from_npz(file_path_a, key_name_a)
    joints_b_np = get_joints_from_npz(file_path_b, key_name_b)

    # 将 numpy 数组转换为 Jittor 张量
    joints_a_jt = jt.array(joints_a_np.astype(np.float32))
    joints_b_jt = jt.array(joints_b_np.astype(np.float32))

    j2j_metric = J2J(joints_a_jt, joints_b_jt)
    return j2j_metric.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算两个NPZ文件中的J2J指标。")
    parser.add_argument(
        "--key_a", 
        type=str, 
        default="joints", 
        help="第一个NPZ文件中关节数据的键名 (默认: 'joints')。"
    )
    parser.add_argument(
        "--key_b", 
        type=str, 
        default="joints", 
        help="第二个NPZ文件中关节数据的键名 (默认: 'joints')。"
    )
    args = parser.parse_args()

    file_a = ""
    file_b = ""

    # --- 示例：创建虚拟NPZ文件用于测试 ---
    # 如果用户没有提供文件，我们创建示例文件
    # --- 实际计算 ---
    print(f"\n正在计算 J2J 指标...")
    print(f"文件 A: '{args.file_a}' (键: '{args.key_a}')")
    print(f"文件 B: '{args.file_b}' (键: '{args.key_b}')")

    try:
        j2j_result = calculate_j2j_from_npz_files(
            args.file_a, 
            args.file_b, 
            key_name_a=args.key_a, 
            key_name_b=args.key_b
        )
        print(f"\n成功计算 J2J 指标！")
        print(f"J2J 指标: {j2j_result:.6f}")
    except (FileNotFoundError, KeyError, ValueError, TypeError, Exception) as e:
        print(f"\n计算 J2J 指标失败: {e}")
    finally:
        # 清理可能创建的示例文件和目录
        if os.path.exists("j2j_samples"):
            print("\n正在清理示例文件和目录...")
            shutil.rmtree("j2j_samples")
        if os.path.exists("__pycache__"): # 清理 Jittor 可能生成的缓存
            shutil.rmtree("__pycache__")
        print("清理完成。")