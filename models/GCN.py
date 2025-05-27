import jittor as jt
from jittor import nn
from jittor import init

from models.gcn_basic_modules import MLP, GCU

class JointPredNet(jt.nn.Module):
    def __init__(self, out_channels, input_normal, arch, aggr='max'):
        super().__init__()
        self.input_normal = input_normal
        self.arch = arch
        self.input_channel = 6 if self.input_normal else 3

        self.gcu_1 = GCU(in_channels=self.input_channel, out_channels=64, aggr=aggr)
        self.gcu_2 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu_3 = GCU(in_channels=256, out_channels=512, aggr=aggr)

        self.mlp_glb = MLP([(64 + 256 + 512), 1024])
        self.mlp_transform = nn.Sequential(
            MLP([1024 + self.input_channel + 64 + 256 + 512, 1024, 256]),
            nn.Dropout(0.7),
            nn.Linear(256, out_channels)
        )

        if self.arch == 'jointnet':
            init.constant_(self.mlp_transform[-1].weight, 0.0)
            init.constant_(self.mlp_transform[-1].bias, 0.0)

    def execute(self, data):
        # data.pos：点的位置  data.x：点的其他属性，比如法向量
        if self.input_normal:
            x = jt.concat([data.pos, data.x], dim=1)
        else:
            x = data.pos

        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch

        x_1 = self.gcu_1(x, tpl_edge_index, geo_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index, geo_edge_index)

        x_4 = self.mlp_glb(jt.concat([x_1, x_2, x_3], dim=1))

        x_global = self.batch_max(x_4, batch) # 找batch中每个样本的最大值
        x_global = jt.reindex(x_global, [x.shape[0], x_global.shape[1]], [
            batch.unsqueeze(1).broadcast(x.shape[0], x_global.shape[1]),
            jt.broadcast_var(jt.arange(x_global.shape[1]), [x.shape[0], x_global.shape[1]])
        ]) # 将全局特征扩展到每个点

        x_5 = jt.concat([x_global, x, x_1, x_2, x_3], dim=1)
        out = self.mlp_transform(x_5)
        if self.arch == 'jointnet':
            out = jt.tanh(out)
        return out

    def batch_max(self, x, batch):
        unique_batch = jt.unique(batch)
        out = []
        for b in unique_batch:
            mask = batch == b
            out.append(x[mask].max(dim=0, keepdims=True))
        return jt.concat(out, dim=0)

class JOINTNET_MASKNET_MEANSHIFT(jt.nn.Module):
    def __init__(self):
        super().__init__()
        self.jointnet = JointPredNet(3, input_normal=False, arch='jointnet', aggr='max')
        self.masknet = JointPredNet(1, input_normal=False, arch='masknet', aggr='max')
        self.bandwidth = jt.Var([0.04])

    def execute(self, data):
        x_offset = self.jointnet(data)
        x_mask_prob_0 = self.masknet(data)
        x_mask_prob = jt.sigmoid(x_mask_prob_0)
        return x_offset, x_mask_prob_0, x_mask_prob, self.bandwidth
