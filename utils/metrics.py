import torch
# import pytorch_lightning as pl
# from pytorch_lightning.metrics import TensorMetric
from torchmetrics import Metric
from typing import Any, Optional
from losses.supervised_losses import *
from losses.unsupervised_losses import *
from losses.common_losses import *
from torchmetrics import MeanMetric
# 计算平均端点误差（EPE），即模型预测场景流与真实场景流之间的
class EPE3D(MeanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        epe3d = torch.norm(pred_flow - gt_flow, dim=2).mean()
        return epe3d
# 计算相对松弛的3D精度，即预测场景流的L2距离小于0.1或相对误差小于0.1的点的百分比。
class Acc3DR(MeanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_relax = (torch.logical_or(l2_norm < 0.1, relative_err < 0.1)).float().mean()
        return acc3d_relax
    
# 计算相对严格的3D精度，即预测场景流的L2距离小于0.05或相对误差小于0.05的点的百分比。
class Acc3DS(MeanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_strict = (torch.logical_or(l2_norm < 0.05, relative_err < 0.05)).float().mean()
        return acc3d_strict
    
# EPE > 0.3 or 相对误差大于0.1 的百分比
class EPE3DOutliers(MeanMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        epe3d_outliers = (torch.logical_or(l2_norm > 0.3, relative_err > 0.1)).float().mean()
        return epe3d_outliers
    
# 预测和真实值的L1 loss误差
class SupervisedL1LossMetric(MeanMetric): 
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.loss = SupervisedL1Loss()
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pc_target, pred_flow, gt_flow)
        return loss_metric


class SmoothnessLossMetric(MeanMetric):
    def __init__(self, smoothness_loss_params, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.loss = SmoothnessLoss(**smoothness_loss_params)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pred_flow)
        return loss_metric

class ChamferLossMetric(MeanMetric):
    def __init__(self, chamfer_loss_params, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.loss = ChamferLoss(**chamfer_loss_params)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pc_target, pred_flow)
        return loss_metric


class SceneFlowMetrics():
    """
    An object of relevant metrics for scene flow.
    """

    def __init__(self, split: str, loss_params: dict, reduce_op: Optional[Any] = None):
        """
        Initializes a dictionary of metrics for scene flow
        keep reduction as 'none' to allow metrics computation per sample.

        Arguments:
            split : a string with split type, should be used to allow logging of same metrics for different aplits
            loss_params: loss configuration dictionary
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                        Defaults to sum.
        """

        self.metrics = {
            split + '_epe3d': EPE3D(),

        }
        if loss_params['loss_type'] == 'sv_l1_reg':
            self.metrics[f'{split}_data_loss'] = SupervisedL1LossMetric()
            self.metrics[f'{split}_smoothness_loss'] = SmoothnessLossMetric(loss_params['smoothness_loss_params'])
        if loss_params['loss_type'] == 'unsup_l1':
            self.metrics[f'{split}_chamfer_loss'] = ChamferLossMetric(loss_params['chamfer_loss_params'])
            self.metrics[f'{split}_smoothness_loss'] = SmoothnessLossMetric(loss_params['smoothness_loss_params'])

        if split in ['test', 'val']:
            self.metrics[f'{split}_acc3dr'] = Acc3DR()
            self.metrics[f'{split}_acc3ds'] = Acc3DS()
            self.metrics[f'{split}_epe3d_outliers'] = EPE3DOutliers()

    def __call__(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flows: list, gt_flow: torch.Tensor) -> dict:
        """
        Compute and scale the resulting metrics

        Arguments:
            pc_source : a tensor containing source point cloud
            pc_target : a tensor containing target point cloud
            pred_flows : list of tensors containing model's predictions
            gt_flow : a tensor containing ground truth labels

        Return:
            A dictionary of copmuted metrics
        """

        result = {}
        for key, metric in self.metrics.items():
            for i, pred_flow in enumerate(pred_flows):
                val = metric(pc_source, pc_target, pred_flow, gt_flow)
                result.update({f'{key}_i#{i}': val})

        return result
