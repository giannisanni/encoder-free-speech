"""Convert training logs ("step N  c0 X  depth Y  Zs/step") to TensorBoard.

Backfill past runs and/or follow a live log:
  python -m gemma_csm.log_to_tb --log ~/m0b.log --run m0b_baseline
  python -m gemma_csm.log_to_tb --log ~/m1_data.log --run m1_multispeaker --follow

Events go to runs_tb/<run>/. Serve with:  tensorboard --logdir runs_tb --bind_all
"""
import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

LINE = re.compile(r"^step (\d+)\s+c0 ([\d.]+)\s+depth ([\d.]+)\s+([\d.]+)s/step")
VAL = re.compile(r"^val (\d+)\s+c0 ([\d.]+)\s+depth ([\d.]+)")


def feed(writer: SummaryWriter, fh, step_offset: int = 0) -> int:
    last = 0
    for line in fh:
        m = LINE.match(line.strip())
        if m:
            step = int(m.group(1)) + step_offset
            writer.add_scalar("loss/c0_semantic", float(m.group(2)), step)
            writer.add_scalar("loss/depth_acoustic", float(m.group(3)), step)
            writer.add_scalar("perf/sec_per_step", float(m.group(4)), step)
            last = step
            continue
        v = VAL.match(line.strip())
        if v:
            step = int(v.group(1)) + step_offset
            writer.add_scalar("val/c0_semantic", float(v.group(2)), step)
            writer.add_scalar("val/depth_acoustic", float(v.group(3)), step)
    writer.flush()
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--step-offset", type=int, default=0,
                   help="add to step numbers (e.g. 32000 for the v2b log)")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs_tb"))
    a = p.parse_args()

    writer = SummaryWriter(str(a.out / a.run))
    with open(a.log, errors="ignore") as fh:
        last = feed(writer, fh, a.step_offset)
        print(f"backfilled to step {last}")
        while a.follow:
            time.sleep(15)
            last = max(last, feed(writer, fh, a.step_offset))


if __name__ == "__main__":
    main()
