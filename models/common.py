import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pointnet2 import pointnet2_utils as pointutils


class FlowEmbedding(nn.Module): # 对局部点云的特征进行编码,它将每个点的位置和特征表示为一个向量，并将其投影到一个低维空间中
    # 生成流嵌入
    def __init__(self, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn=True, use_instance_norm=False):
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func == 'concat':
            last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if use_instance_norm:
                self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
            else:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            pos1: (batch_size, 3, npoint) 点云位置
            pos2: (batch_size, 3, npoint) 
            feature1: (batch_size, channel, npoint) 点云特征
            feature2: (batch_size, channel, npoint)
        Output:
            pos1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, N, C = pos1_t.shape
        if self.knn:
            # 在pos1中选择与pos2中距离最近的S个点
            # idx: [B, N, S] N个来自pos1的点有S个最近的来自pos2的点
            dist, idx = pointutils.knn(self.nsample, pos1_t, pos2_t) 
            # 相当于每个点最近的邻居的索引，idx[:, :, 0]会降一维，unseq(2)把第2维升，从[B,N]变成[B,N,1]
            # repeat是指定维度重复次数，0,1维度重复一次，2维重复S次，[B,N,1] => [B,N,S]
            tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.nsample).to(idx.device)
            # dist > self.radius，生成mask，距离超过radius的，让它等于最小的距离
            # 将半径为 radius 的球内部点的个数少于 nsample 个的情况进行处理。
            idx[dist > self.radius] = tmp_idx[dist > self.radius]
        else:
            # If the ball neighborhood points are less than nsample,
            # than use the knn neighborhood points
            # 在球里面的idx和数量，cnt 是 [B,N]
            idx, cnt = pointutils.ball_query(self.radius, self.nsample, pos2_t, pos1_t)
            _, idx_knn = pointutils.knn(self.nsample, pos1_t, pos2_t)
            # [B,N] => [B,N,1] => [B,N,S]
            cnt = cnt.view(B, -1, 1).repeat(1, 1, self.nsample)
            idx = idx_knn[cnt > (self.nsample - 1)]

        # 其中pos2是形状为[B, 3, N]的tensor，idx是形状为[B, N, S]的tensor
        # 将pos2按照idx中的索引分组，相当于记录pos1的点，对应的pos2的点
        pos2_grouped = pointutils.grouping_operation(pos2, idx)  # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B, 3, N, S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)  # [B, C, N, S]
        if self.corr_func == 'concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, self.nsample)], dim=1)

        feat1_new = torch.cat([pos_diff, feat_diff], dim=1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        return pos1, feat1_new


class PointNetSetAbstraction(nn.Module):
    # 分层抽象
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all,
                 return_fps=False, use_xyz=True, use_act=True, act=F.relu, mean_aggr=False, use_instance_norm=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        self.act = act
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if use_instance_norm:
                self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
            else:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if group_all:
            self.queryandgroup = pointutils.GroupAll(self.use_xyz)
        else:
            self.queryandgroup = pointutils.QueryAndGroup(radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N] 点云位置
            points: input points data, [B, D, N] 点云特征
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        # FPS：随机选择一个点 $p_0$ 作为第一个采样点。
        # 对于点云中的每一个点 $p_i$，计算其到所有已选采样点的距离 $d_i$，并取其中最小的距离 $d_{min}$。
        # 将具有最大距离 $d_{max}$ 的点 $p_{max}$ 加入采样点集合，并将 $p_{max}$ 作为下一轮计算距离的起点。
        # 重复步骤 2 和 3，直到采样点数量达到设定值或者点云中的所有点都已被选为采样点。
        if (self.group_all == False) and (self.npoint != -1):
            if fps_idx == None:
                fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        else:
            new_xyz = xyz
        new_points, _ = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_act:
                bn = self.mlp_bns[i]
                new_points = self.act(bn(conv(new_points)))
            else:
                new_points = conv(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1) # 对最后一个维度求平均 [B, C+D, npoint, nsample] => [B, C+D, npoint] 一个点对应的领域里面的点的平均值
        else:
            new_points = torch.max(new_points, -1)[0] # 领域里的最远点

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points


class PointNetFeaturePropogation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropogation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.apply_mlp = mlp is not None
        last_channel = in_channel
        if self.apply_mlp:
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        # 上采样点云特征
        """
        Input:
            pos1: input points position data, [B, C, N]
            pos2: sampled input points position data, [B, C, S]
            feature1: input points data, [B, D, N]
            feature2: input points data, [B, D, S]
        Return:
            feat_new: upsampled points data, [B, D', N]
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        # 3-NN 在采样点云上找到对于每个原始点云位置最近的三个采样点 然后根据这三个点的距离权重进行插值，生成每个原始点云位置上的特征。
        dists, idx = pointutils.three_nn(pos1_t, pos2_t)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1, keepdim=True)  # [B,N,3]
        interpolated_feat = torch.sum(pointutils.grouping_operation(feature2, idx) * weight.view(B, 1, N, 3),
                                      dim=-1)  # [B,C,N,3] => [B,C,N]

        if feature1 is not None:
            feat_new = torch.cat([interpolated_feat, feature1], 1)
        else:
            feat_new = interpolated_feat

        if self.apply_mlp:
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                feat_new = F.relu(bn(conv(feat_new)))
        return feat_new
