"""通过 CPU 或 Slurm GPU 验证端到端训练流程；示例不是科学重建模型。"""

import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset
from models import build_model
from utils.runtime import (PROJECT_ROOT, arguments, atomic_write, checksum, finish_record,
                           initialize, load_config, start_record, storage_root)


def main():
    args = arguments()
    config = load_config(args)
    rank, world, device = initialize(args.device, config["seed"])
    run = None
    started = time.perf_counter()
    try:
        if rank == 0:
            run, checkpoint_dir, logger, record = start_record(config, args, "train", device, world)
        dataset, source = build_dataset(config, PROJECT_ROOT, storage_root())
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=config["seed"]) if world > 1 else None
        generator = torch.Generator().manual_seed(config["seed"] + rank)
        loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], sampler=sampler,
                            shuffle=sampler is None, num_workers=config["training"]["num_workers"], generator=generator)
        model = build_model(config["model"]).to(device)
        if world > 1:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        if rank == 0:
            record.update(dataset_samples=len(dataset), model_parameter_count=sum(p.numel() for p in model.parameters()),
                          dataset_path=str(source) if source else None, dataset_sha256=checksum(source) if source else None)
            record["epochs"] = []
        for epoch in range(config["training"]["epochs"]):
            if sampler:
                sampler.set_epoch(epoch)
            model.train()
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失不是有限数值。")
                loss.backward()
                optimizer.step()
                totals[0] += loss.detach().double() * y.numel()
                totals[1] += y.numel()
            if world > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            mse = (totals[0] / totals[1]).item()
            if rank == 0:
                record["epochs"].append({"epoch": epoch + 1, "training_mse": mse})
                logger.info("epoch=%d training_mse=%.8f", epoch + 1, mse)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if rank == 0:
            base_model = model.module if world > 1 else model
            checkpoint = checkpoint_dir / "last.pt"
            payload = {"model_state": {key: value.detach().cpu() for key, value in base_model.state_dict().items()},
                       "model_config": config["model"], "seed": config["seed"], "git_commit": record["git"]["commit"]}
            atomic_write(checkpoint, lambda tmp: torch.save(payload, tmp))
            record.update(checkpoint=str(checkpoint), checkpoint_sha256=checksum(checkpoint), training_mse=mse)
            finish_record(run, logger, record, time.perf_counter() - started)
    except Exception as error:
        if rank == 0 and run is not None:
            finish_record(run, logger, record, time.perf_counter() - started, error)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
