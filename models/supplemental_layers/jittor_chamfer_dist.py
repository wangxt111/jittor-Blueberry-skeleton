import jittor as jt
from jittor import nn

def chamfer_distance_with_average(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size [1, N, D]
    :param p2: size [1, M, D]
    :return: average Chamfer Distance between p1 and p2
    '''
    assert p1.shape[0] == 1 and p2.shape[0] == 1
    assert p1.shape[2] == p2.shape[2]

    N = p1.shape[1]
    M = p2.shape[1]

    # 扩展 p1 和 p2 为 [N, M, D]
    p1_exp = p1.broadcast((1, N, M, p1.shape[2]))
    p2_exp = p2.broadcast((1, M, N, p2.shape[2])).permute(0, 2, 1, 3)

    # 计算欧几里得距离
    dist = p1_exp - p2_exp
    dist_norm = jt.norm(dist, p=2, dim=3)  # [1, N, M]

    dist1 = jt.min(dist_norm, dim=2)[0]  # [1, N]
    dist2 = jt.min(dist_norm, dim=1)[0]  # [1, M]

    loss = 0.5 * (jt.mean(dist1) + jt.mean(dist2))
    return loss
