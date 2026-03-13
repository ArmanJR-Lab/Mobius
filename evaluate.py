# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "onnxruntime",
#     "pandas",
#     "tqdm",
# ]
# ///
"""
Fixed evaluation harness for the Mobius controller optimization loop.

Loads Controller from controller.py, runs it against N shuffled segments
using the TinyPhysics simulator, and prints a greppable cost summary.

Do NOT modify this file — it is part of the fixed infrastructure.
"""

import argparse
import importlib.util
import logging
import multiprocessing
import time
from functools import partial
from pathlib import Path

import numpy as np

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, download_dataset, DATASET_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "tinyphysics.onnx"
CONTROLLER_PATH = Path(__file__).resolve().parent / "controller.py"

DEFAULT_NUM_SEGS = 1000
SHUFFLE_SEED = 42
NUM_WORKERS = 16


def load_controller_class():
    """Dynamically load Controller from controller.py."""
    spec = importlib.util.spec_from_file_location("controller", CONTROLLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


def run_single_segment(data_path: str, model_path: str) -> dict:
    """Run one segment end-to-end in a worker process."""
    ControllerClass = load_controller_class()
    model = TinyPhysicsModel(str(model_path))
    controller = ControllerClass()
    sim = TinyPhysicsSimulator(model, data_path, controller=controller)
    return sim.rollout()


def main():
    parser = argparse.ArgumentParser(description="Evaluate controller on driving segments")
    parser.add_argument("--num-segs", type=int, default=DEFAULT_NUM_SEGS,
                        help=f"Number of segments to evaluate (default: {DEFAULT_NUM_SEGS})")
    args = parser.parse_args()

    # Auto-download dataset if missing
    if not DATASET_PATH.exists() or not any(DATASET_PATH.iterdir()):
        log.info("Dataset not found — downloading...")
        download_dataset()
        log.info("Dataset downloaded to %s", DATASET_PATH)

    # Gather and shuffle segments with fixed seed
    all_segments = sorted(DATASET_PATH.glob("*.csv"))
    if not all_segments:
        log.error("No CSV files found in %s", DATASET_PATH)
        raise SystemExit(1)

    rng = np.random.RandomState(SHUFFLE_SEED)
    indices = rng.permutation(len(all_segments))
    selected = [str(all_segments[i]) for i in indices[:args.num_segs]]

    log.info("Evaluating %d / %d segments with %d workers", len(selected), len(all_segments), NUM_WORKERS)
    t0 = time.time()

    worker_fn = partial(run_single_segment, model_path=str(MODEL_PATH))
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(pool.imap_unordered(worker_fn, selected, chunksize=10))

    elapsed = time.time() - t0

    # Aggregate costs
    lataccel_costs = [r["lataccel_cost"] for r in results]
    jerk_costs = [r["jerk_cost"] for r in results]
    total_costs = [r["total_cost"] for r in results]

    mean_lataccel = np.mean(lataccel_costs)
    mean_jerk = np.mean(jerk_costs)
    mean_total = np.mean(total_costs)

    log.info("Evaluation complete in %.1fs", elapsed)

    # Greppable summary block
    print("---")
    print(f"total_cost:    {mean_total:.6f}")
    print(f"lataccel_cost: {mean_lataccel:.6f}")
    print(f"jerk_cost:     {mean_jerk:.6f}")
    print(f"num_segments:  {len(selected)}")
    print(f"elapsed_secs:  {elapsed:.1f}")


if __name__ == "__main__":
    main()
