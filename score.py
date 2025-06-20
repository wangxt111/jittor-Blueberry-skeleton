import argparse
import jittor as jt
import numpy as np
from models.metrics import J2J

# 定义新的 compute_cd_j2j_loss 函数
# pred_joints, gt_joints: [B, 22, 3]
def compute_cd_j2j_loss(pred_joints, gt_joints):
    # Jittor 的 norm 函数计算的是 L2 范数（欧几里得距离）
    # dim=2 表示在最后一个维度（坐标维度）上计算范数
    distances = jt.norm(pred_joints - gt_joints, dim=2)  # 结果形状为 [B, 22]
    # 对所有距离求平均
    cd_j2j = jt.mean(distances)  # 标量
    return cd_j2j

def transform(vertices, joints):
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    center = (min_vals + max_vals) / 2
    scale = np.max(max_vals - min_vals) / 2
    normalized_joints = (joints - center) / scale
    return normalized_joints

def get_files(file_list: str):
    with open("data/" + file_list, "r") as f:
        lines = f.readlines()
    files = [line.strip()[6:-4] for line in lines]
    return files

def get_file_score(file_path: str, skeleton_model_name: str, skin_model_name: str):
    ref = np.load("data/train/" + file_path + ".npz", allow_pickle=True)
    ref_vertices = ref["vertices"]
    ref_joints = ref["joints"]
    ref_skin = ref["skin"]
    ref_joints = transform(ref_vertices, ref_joints)

    pre_joints = np.load(
        "predict/" + skeleton_model_name + "/" + file_path + "/predict_skeleton.npy", allow_pickle=True
    )
    pre_skin = np.load(
        "predict/" + skeleton_model_name + "/" + file_path + "/predict_skin.npy", allow_pickle=True
    )

    print(f"{file_path} joints: ref_joints.shape={ref_joints.shape}, pre_joints.shape={pre_joints.shape}")
    print(f"{file_path} skin: ref_skin.shape={ref_skin.shape}, pre_skin.shape={pre_skin.shape}")

    # 确保 pred_joints_jt 和 ref_joints_jt 总是被定义
    # 首先将 numpy 数组转换为 Jittor 张量
    ref_joints_jt_raw = jt.array(ref_joints)
    pre_joints_jt_raw = jt.array(pre_joints)

    # 然后检查它们的形状，并在必要时添加批次维度
    if len(ref_joints_jt_raw.shape) == 2: # 假设形状是 [num_joints, 3]，例如 [22, 3] (单帧)
        ref_joints_jt = ref_joints_jt_raw.unsqueeze(0) # 添加一个批次维度: [1, 22, 3]
    elif len(ref_joints_jt_raw.shape) == 3: # 假设形状是 [num_frames, num_joints, 3]，例如 [B, 22, 3]
        ref_joints_jt = ref_joints_jt_raw
    else:
        raise ValueError(f"Unexpected ref_joints shape: {ref_joints_jt_raw.shape}. Expected 2 or 3 dimensions.")

    if len(pre_joints_jt_raw.shape) == 2: # 假设形状是 [num_joints, 3]
        pre_joints_jt = pre_joints_jt_raw.unsqueeze(0) # 添加一个批次维度: [1, 22, 3]
    elif len(pre_joints_jt_raw.shape) == 3: # 假设形状是 [num_frames, num_joints, 3]
        pre_joints_jt = pre_joints_jt_raw
    else:
        raise ValueError(f"Unexpected pre_joints shape: {pre_joints_jt_raw.shape}. Expected 2 or 3 dimensions.")

    # 计算 J2J 损失
    # 传递预测关节点在前，真实关节点在后
    cdj2j = compute_cd_j2j_loss(pre_joints_jt, ref_joints_jt).item()
    # cdj2j = J2J(jt.array(ref_joints), jt.array(pre_joints)).item()

    # 计算 Skin L1 损失
    skinl1 = np.mean(np.abs(ref_skin - pre_skin))
    return cdj2j, skinl1

def get_score(skeleton_model_name: str, skin_model_name: str):
    files = get_files("val_list.txt")
    cdj2js = []
    skinl1s = []
    for file in files:
        cdj2j, skinl1 = get_file_score(file, skeleton_model_name, skin_model_name)
        cdj2js.append(cdj2j)
        skinl1s.append(skinl1)
        print(file, "cdj2j: ", cdj2j, "skinl1: ", skinl1)

    cdj2j_mean = np.mean(cdj2js)
    skinl1_mean = np.mean(skinl1s)
    
    score = 1 / min(cdj2j_mean, 0.01) * (0.5 * (1 - 20 * min(skinl1_mean, 0.05)))
    print("cdj2j: ", cdj2j_mean, "skinl1: ", skinl1_mean, "score: ", score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton_model_name", type=str, default="pct")
    parser.add_argument("--skin_model_name", type=str, default="pct")
    args = parser.parse_args()
    skeleton_model_name = args.skeleton_model_name
    skin_model_name = args.skin_model_name
    
    get_score(skeleton_model_name, skin_model_name)