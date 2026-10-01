"""确定性的合成数据示例；真实轨迹和流场的数据加载器需另行实现。"""
import torch
from torch.utils.data import TensorDataset


def build_dataset(config, project_root, storage_root):
    data, model = config["dataset"], config["model"]
    source = None
    if data["kind"] == "synthetic":
        generator = torch.Generator().manual_seed(data["seed"])
        x = torch.randn(data["samples"], model["input_features"], generator=generator)
        weights = torch.randn(model["input_features"], model["output_features"], generator=generator)
        y = x @ weights
    else:
        raise ValueError("当前工作流示例仅实现 dataset.kind=synthetic。")
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("数据必须是长度相同且非空的二维 x / y 数组。")
    if x.shape[1] != model["input_features"] or y.shape[1] != model["output_features"]:
        raise ValueError("数据维度与模型配置不匹配。")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("数据包含 NaN / Inf。")
    return TensorDataset(x, y), source
