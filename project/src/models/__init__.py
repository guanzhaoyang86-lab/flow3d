"""工作流验证模型；实际科学模型应在这里单独实现。"""

from torch import nn


def build_model(config):
    if config["name"] != "synthetic_mlp":
        raise ValueError("当前示例只实现 synthetic_mlp，不是生成式流场重建模型。")
    return nn.Sequential(
        nn.Linear(config["input_features"], config["hidden_features"]),
        nn.ReLU(),
        nn.Linear(config["hidden_features"], config["output_features"]),
    )
