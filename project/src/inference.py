"""示例推理输出 NPZ；只接收本工作流产生、来源可信的检查点。"""

import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.runtime import (PROJECT_ROOT, arguments, atomic_write, finish_record,
                           initialize, load_config, start_record, storage_root)


def main():
    args = arguments(inference=True)
    config = load_config(args)
    rank, world, device = initialize(args.device, config["seed"])
    run = None
    started = time.perf_counter()
    try:
        if world != 1:
            raise ValueError("示例推理仅支持单进程；请分配一张 GPU。训练支持多 GPU DDP。")
        run, _, logger, record = start_record(config, args, "inference", device, world)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload["model_config"] != config["model"]:
            raise ValueError("模型配置与检查点不一致；请使用训练时保存的配置。")
        model = build_model(config["model"]).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        dataset, _ = build_dataset(config, PROJECT_ROOT, storage_root())
        predictions, targets = [], []
        with torch.inference_mode():
            for x, y in DataLoader(dataset, batch_size=config["training"]["batch_size"]):
                predictions.append(model(x.to(device)).cpu().numpy())
                targets.append(y.numpy())
        predictions, targets = np.concatenate(predictions), np.concatenate(targets)
        output = run / "predictions.npz"
        def save_output(path):
            with path.open("wb") as handle:
                np.savez_compressed(handle, predictions=predictions, targets=targets)
        atomic_write(output, save_output)
        record.update(predictions=str(output), samples=len(dataset),
                      evaluation_mse=float(np.mean((predictions - targets) ** 2)),
                      evaluation_note="使用配置指定的数据；默认即合成训练数据，不代表泛化性能。")
        finish_record(run, logger, record, time.perf_counter() - started)
    except Exception as error:
        if run is not None:
            finish_record(run, logger, record, time.perf_counter() - started, error)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
