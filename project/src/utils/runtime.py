"""小型示例运行工具；每次运行独立保存，支持重新执行，不提供断点续训。"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def arguments(inference=False):
    parser = argparse.ArgumentParser(description="合成 MLP 工作流验证示例（非研究模型）")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true", help="合成数据缩为 32 条，训练 1 轮")
    if inference:
        parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def load_config(args):
    path = Path(args.config).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "model" not in config:
        model_path = (path.parent / config["model_config"]).resolve()
        with model_path.open(encoding="utf-8") as handle:
            config["model"] = yaml.safe_load(handle)
    config["source_config"] = str(path)
    if args.smoke_test:
        config["training"]["epochs"] = 1
        if config["dataset"]["kind"] == "synthetic":
            config["dataset"]["samples"] = min(32, config["dataset"]["samples"])
    config["execution"] = {"device": args.device, "smoke_test": args.smoke_test}
    if config["training"]["epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("epochs 和 batch_size 必须为正数。")
    return config


def initialize(device_name, seed):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_name == "cuda":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("GPU 训练 / 推理只允许在 Slurm 作业中运行；本地请用 --device cpu。")
        if not torch.cuda.is_available():
            raise RuntimeError("Slurm 作业内 PyTorch 未检测到 CUDA；检查 GPU 分配及 PyTorch 安装。")
        if os.environ.get("SLURM_GPUS_ON_NODE") == "0" or local_rank >= torch.cuda.device_count():
            raise RuntimeError("GPU 分配不足；每个 torchrun 进程需要一张 GPU。")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return rank, world, device


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="pending-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"))


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def storage_root():
    value = os.environ.get("FLOW3D_ROOT")
    return Path(value).expanduser().resolve() if value else None


def git_info():
    def query(*options):
        try:
            return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *options], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "commit": os.environ.get("FLOW3D_COMMIT") or query("rev-parse", "HEAD"),
        "working_tree_status": query("status", "--porcelain", "--untracked-files=normal"),
    }


def start_record(config, args, mode, device, world):
    name = re.sub(r"[^\w.-]+", "_", config["experiment_name"])
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%d_%H%M%S}_{name}_{mode}_{uuid4().hex[:8]}"
    root = storage_root()
    run = (root / "results" / "experiments" if root else PROJECT_ROOT / "experiments") / run_id
    run.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = root / "checkpoints" / run_id if root else run / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if root:
        try:
            (run / "checkpoint").symlink_to(checkpoint_dir, target_is_directory=True)
        except OSError:
            (run / "checkpoint_location.txt").write_text(str(checkpoint_dir), encoding="utf-8")
    logger = logging.getLogger("flow3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(run / "log.txt", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    atomic_write(run / "config.yaml", lambda tmp: tmp.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"))
    command = shlex.join([sys.executable, *sys.argv])
    (run / "command.txt").write_text(command + "\n", encoding="utf-8")
    try:
        packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        packages = f"# pip freeze unavailable: {error}\n"
    (run / "environment.txt").write_text(packages, encoding="utf-8")
    record = {
        "status": "running", "mode": mode, "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "demo_only": config["model"]["name"] == "synthetic_mlp",
        "seed": config["seed"], "model": config["model"],
        "dataset_version": config["dataset"]["version"],
        "device": str(device), "world_size": world,
        "gpu_type": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_submit_command": os.environ.get("FLOW3D_SUBMIT_COMMAND"),
        "git": git_info(), "run_directory": str(run), "checkpoint_directory": str(checkpoint_dir),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda, "platform": platform.platform()},
        "reproducibility": "记录支持按相同配置重新运行；不提供断点续训，也不保证跨硬件逐位一致。",
    }
    if mode == "inference":
        record["input_checkpoint"] = str(Path(args.checkpoint).resolve())
        record["input_checkpoint_sha256"] = checksum(args.checkpoint)
    write_json(run / "results.json", record)
    logger.info("合成 MLP 工作流示例；实验目录：%s", run)
    return run, checkpoint_dir, logger, record


def finish_record(run, logger, record, seconds, error=None):
    record.update(status="failed" if error else "completed", elapsed_seconds=seconds, finished_utc=datetime.now(timezone.utc).isoformat())
    if error:
        record["error"] = str(error)
        logger.exception("运行失败")
    write_json(run / "results.json", record)
    logger.info("运行状态：%s；耗时：%.3f 秒", record["status"], seconds)
