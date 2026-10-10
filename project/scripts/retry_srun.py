#!/usr/bin/env python3
"""Retry a failed single-task Slurm launch, never an already-started experiment.

Only the known interconnect launch error is eligible. Retries stay within the
existing allocation and time limit; no jobs are requeued or submitted here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time


LAUNCH_ERROR = re.compile(
    r"^srun: error: task 0 launch failed: Error configuring interconnect\s*$"
)
CANCEL_ERROR = re.compile(r"CANCELLED|CANCELED|DUE TO TIME LIMIT", re.IGNORECASE)


def payload_exec(arguments: list[str]) -> int:
    marker, separator, *command = arguments
    if separator != "--" or not command:
        raise ValueError("invalid payload command")
    # This must be on shared storage, not node-local /tmp. Once this exists,
    # retrying is forbidden even if the payload prints a matching error itself.
    Path(marker).write_text("payload started\n", encoding="utf-8")
    if "FLOW3D_EVALUATE_BUDGET_SECONDS" in os.environ:
        elapsed = max(0, math.ceil(time.time() - float(os.environ["FLOW3D_LAUNCH_STARTED_AT"])))
        remaining = int(os.environ["FLOW3D_EVALUATE_BUDGET_SECONDS"]) - elapsed
        if remaining <= 0:
            print("Evaluation time budget exhausted during Slurm launch.", file=sys.stderr)
            return 75
        os.environ["FLOW3D_EVALUATE_BUDGET_SECONDS"] = str(remaining)
    os.execvp(command[0], command)
    return 127  # exec does not return on POSIX.


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments and arguments[0] == "_exec":
        return payload_exec(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retries", type=int, default=os.environ.get("FLOW3D_LAUNCH_RETRIES", "3"))
    parser.add_argument("--delay", type=float, default=os.environ.get("FLOW3D_LAUNCH_RETRY_DELAY", "30"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not 0 <= args.retries <= 3:
        parser.error("--retries must be between 0 and 3 (additional attempts)")
    if not math.isfinite(args.delay) or not 0 <= args.delay <= 300:
        parser.error("--delay must be between 0 and 300 seconds")
    if not command:
        parser.error("missing payload command")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job):
        parser.error("run inside a Slurm allocation (SLURM_JOB_ID required)")
    root = Path(os.environ.get("FLOW3D_ROOT", ""))
    if not root.is_absolute():
        parser.error("FLOW3D_ROOT must be an absolute shared storage path")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record_dir = Path(tempfile.mkdtemp(prefix=f"launch-{job}-", dir=logs))
    print(f"Launch retry records: {record_dir}", flush=True)
    (record_dir / "launcher.json").write_text(json.dumps({
        "job_id": job, "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "science_commit": os.environ.get("FLOW3D_COMMIT"),
        "launcher_commit": os.environ.get("FLOW3D_LAUNCHER_COMMIT"),
        "plan_sha256": os.environ.get("FLOW3D_PLAN_SHA256"),
        "command": command, "max_extra_retries": args.retries,
        "retry_delay_seconds": args.delay,
    }, indent=2), encoding="utf-8")

    stop = threading.Event()
    cancelled = 0
    child: subprocess.Popen | None = None

    def cancel(signum, _frame):
        nonlocal cancelled
        cancelled = signum
        stop.set()
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    launch_environment = {**os.environ, "FLOW3D_LAUNCH_STARTED_AT": str(time.time())}
    try:
        for attempt in range(1, args.retries + 2):
            if cancelled:
                return 128 + cancelled
            marker = record_dir / f"attempt-{attempt}.started"
            launch = ["srun", "--ntasks=1", "--gpu-bind=none", sys.executable,
                      str(Path(__file__).resolve()), "_exec", str(marker), "--", *command]
            matched_error = cancelled_message = False
            print(f"Slurm launch attempt {attempt}/{args.retries + 1}", flush=True)
            with (record_dir / f"attempt-{attempt}.stderr").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(launch, stderr=subprocess.PIPE, text=True,
                                         encoding="utf-8", errors="replace", env=launch_environment)
                if cancelled:
                    cancel(cancelled, None)
                assert child.stderr is not None
                for line in child.stderr:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                    log.write(line)
                    log.flush()
                    matched_error |= bool(LAUNCH_ERROR.fullmatch(line.rstrip("\r\n")))
                    cancelled_message |= bool(CANCEL_ERROR.search(line))
                child.stderr.close()
                status = child.wait()
            started = marker.exists()
            retry = (status != 0 and 0 < status < 128 and matched_error and not started
                     and not cancelled and not cancelled_message and attempt <= args.retries)
            with (record_dir / "attempts.jsonl").open("a", encoding="utf-8") as record:
                record.write(json.dumps({"attempt": attempt, "exit_code": status,
                                         "payload_started": started, "will_retry": retry}) + "\n")
            if cancelled:
                return 128 + cancelled
            if not retry:
                return status if status >= 0 else 128 - status
            print(f"Interconnect launch failed before payload start; retrying in {args.delay:g}s "
                  "within the same allocation.", file=sys.stderr, flush=True)
            if stop.wait(args.delay):
                return 128 + cancelled
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 1


if __name__ == "__main__":
    sys.exit(main())
