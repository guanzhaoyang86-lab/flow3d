"""实际执行 CPU 训练、保存配置重放、推理，并验证实验记录。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class SmokeTest(unittest.TestCase):
    def execute(self, entry, *args):
        before = set((PROJECT / "experiments").glob("*/results.json"))
        environment = os.environ.copy()
        for key in ("FLOW3D_ROOT", "SLURM_JOB_ID", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
            environment.pop(key, None)
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, f"src/{entry}.py", *args], cwd=PROJECT,
                                env=environment, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        paths = set((PROJECT / "experiments").glob("*/results.json")) - before
        self.assertEqual(len(paths), 1)
        path = paths.pop()
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "completed")
        for filename in ("config.yaml", "command.txt", "environment.txt", "log.txt"):
            self.assertTrue((path.parent / filename).is_file(), filename)
        return path.parent, record

    def test_train_replay_inference(self):
        run, training = self.execute("train", "--config", "configs/train.yaml", "--device", "cpu", "--smoke-test")
        self.assertTrue(Path(training["checkpoint"]).is_file())
        self.assertEqual(training["dataset_version"], "synthetic-linear-v1")
        _, replay = self.execute("train", "--config", str(run / "config.yaml"), "--device", "cpu")
        self.assertEqual(training["training_mse"], replay["training_mse"])
        _, inference = self.execute("inference", "--config", str(run / "config.yaml"), "--device", "cpu",
                                    "--checkpoint", training["checkpoint"])
        self.assertEqual(inference["input_checkpoint_sha256"], training["checkpoint_sha256"])
        self.assertTrue(Path(inference["predictions"]).is_file())
        self.assertEqual(inference["samples"], training["dataset_samples"])


if __name__ == "__main__":
    unittest.main()
