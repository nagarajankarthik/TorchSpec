# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pipeline training loop: main training loop with sync training and async inference."""

import re
import shutil
import tempfile
import time
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from torchspec.training.data_fetcher import TrainSample
from torchspec.utils.logging import get_tb_writer, logger


def cleanup_mooncake_data(sample: TrainSample, store) -> None:
        """Remove data from mooncake store to release buffer space."""
        shapes = sample.tensor_shapes or {}
        has_lhs = "last_hidden_states" in shapes
        has_target = "target" in shapes

        store.remove_eagle3_tensors(
            sample.mooncake_key,
            has_last_hidden_states=has_lhs,
            has_target=has_target,
        )

def _write_training_metrics(metrics: dict, train_step: int, inference_step: int) -> None:
    loss = metrics.get("train/avg_loss")
    if isinstance(loss, (int, float)):
        logger.info(f"TRAIN_STEP step={train_step} loss={loss:.6f}")

    if getattr(wandb, "run", None) is not None:
        wandb.log(metrics)

    tb_writer = get_tb_writer()
    if tb_writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                scalar_step = inference_step if key.startswith("inference/") else train_step
                tb_writer.add_scalar(key, value, scalar_step)


def _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines):
    """Sync draft model weights to inference engines (decode mode only)."""
    weight_sync_enabled = getattr(args, "decode_weight_sync_enabled", False)
    weight_sync_interval = getattr(args, "decode_weight_sync_interval", 500)
    if not (
        getattr(args, "train_with_decode", False)
        and weight_sync_enabled
        and inference_engines
        and weight_sync_interval > 0
        and completed_steps > 0
        and completed_steps % weight_sync_interval == 0
    ):
        return

    # NOTE: uses local tmp dir; for multi-node, ensure output_dir is on shared filesystem.
    tmp_dir = tempfile.mkdtemp(prefix="draft_weight_sync_")
    try:
        logger.info(f"Step {completed_steps}: saving draft model to {tmp_dir}")
        train_group.save_draft_model_for_serving(tmp_dir)

        logger.info(f"Step {completed_steps}: updating {len(inference_engines)} engine(s)")
        update_results = ray.get(
            [engine.update_weights_from_disk.remote(tmp_dir) for engine in inference_engines]
        )
        for i, res in enumerate(update_results):
            log_fn = logger.info if res.get("success") else logger.warning
            log_fn(f"Engine {i}: success={res.get('success')}, message={res.get('message')}")
    except Exception:
        logger.exception(f"Step {completed_steps}: weight sync failed, skipping")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp dir {tmp_dir}")


def _is_save_interval_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _cleanup_old_checkpoints(checkpoint_dir: str | None, max_checkpoints: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_checkpoints`.

    Checkpoint directories are named ``iter_NNNNNNN`` where N is the step number.
    The ``latest_checkpointed_iteration.txt`` and ``best_*`` files are preserved.
    """
    if not checkpoint_dir or max_checkpoints <= 0:
        return

    base_dir = Path(checkpoint_dir).expanduser()
    if not base_dir.exists():
        return

    # Find all iter_* directories, sorted by step number
    iter_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and re.match(r"iter_\d+", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if len(iter_dirs) <= max_checkpoints:
        return

    # Delete oldest checkpoints, keep the newest max_checkpoints
    to_delete = iter_dirs[: len(iter_dirs) - max_checkpoints]
    for old_dir in to_delete:
        logger.info(f"Removing old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except OSError as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")



def _safe_training_cleanup(
    args, inference_manager, controller = None, mooncake_master = None, mooncake_store = None, inference_future = None, inference_engines=None
) -> None:
    """Best-effort teardown for inference manager and mooncake master actor."""
    if inference_manager is not None:
        try:
            inference_manager.stop()
        except Exception as exc:
            logger.warning(f"Failed to stop inference manager: {exc}")
        if inference_future is not None:
            try:
                ray.get(inference_future)
            except Exception as exc:
                logger.warning(
                    f"Inference manager run loop exited with error during cleanup: {exc}"
                )

    if inference_engines:
        logger.info(f"Shutting down {len(inference_engines)} inference engine(s)...")
        shutdown_refs = []
        for engine in inference_engines:
            try:
                shutdown_refs.append(engine.shutdown.remote())
            except Exception as exc:
                logger.warning(f"Failed to initiate engine shutdown: {exc}")
        for ref in shutdown_refs:
            try:
                ray.get(ref, timeout=30)
            except Exception as exc:
                logger.warning(f"Engine shutdown timed out or failed: {exc}")

    if mooncake_store is not None:
        for leftover in controller.drain_pool():
            cleanup_mooncake_data(leftover, mooncake_store)

    if mooncake_master is not None:
        try:
            mooncake_master.shutdown()
        except Exception as exc:
            logger.warning(f"Failed to shutdown mooncake master: {exc}")

    if mooncake_store is not None:
        try:
            mooncake_store.close()
        except Exception as exc:
            logger.warning(f"Failed to shutdown mooncake store: {exc}")


def training_loop(
    args,
    controller,
    inference_manager,
    mooncake_store,
    dataset_size=None,
    eval_dataset_size=None,
):
    """
    In this branch, this function has been modified to run inference only. 

    Hence, terms like completed_steps and num_steps refer to the number of times 
    `try_dispatch_batch` is successfully executed by the training controller.

    They have no connection to the number of optimizer steps performed by the trainers.

    Evaluation is temporarily disabled for now.

    Training is synchronous - waits for each step to complete.
    Inference runs in background, continuously producing data.

    Each optimizer step (with draft_accumulation_steps dispatches):
      1. Controller dispatches per_dp_rank_batch_size * dp_size samples, accumulation_steps times
      2. Each DP rank receives per_dp_rank_batch_size * accumulation_steps samples total
      3. train_from_queue(num_batches=accumulation_steps) processes all micro-batches
      4. Optimizer steps after the last micro-batch

    completed_steps counts optimizer steps (consistent with lr_total_steps).

    Args:
        args: Configuration arguments.
        controller: AsyncTrainingController ray actor handle (dataset already loaded).
        inference_manager: AsyncInferenceManager ray actor handle.
        train_group: Training group with set_train_queues method.
        dataset_size: Number of training samples. If None, queried from controller.
        eval_dataset_size: Number of eval samples. If None, queried from controller. 0 means no eval.
    """
    if dataset_size is None:
        dataset_size = controller.get_dataset_size()
    if dataset_size <= 0:
        raise ValueError(
            f"Training dataset size is {dataset_size}. "
            f"Ensure controller.load_dataset() was called before run_training_loop()."
        )

    steps_per_epoch = dataset_size // controller.dispatch_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    num_epochs = getattr(args, "num_epochs", 1)
    num_steps = num_epochs * steps_per_epoch

    def _pipeline_idle() -> bool:
        st = inference_manager.get_status()
        return (st["prompt_buffer_size"] == 0
                and st["pending_tasks"] == 0
                and controller.get_pool_size() == 0)

    startup_deadline = time.monotonic() + 60
    while _pipeline_idle():
        if time.monotonic() > startup_deadline:
            raise RuntimeError("inference manager never picked up any prompts")
        time.sleep(0.1)

    # Submit training data AFTER eval hs generation so that training prompts don't
    # leak into the inference pipeline during eval.
    # Resume is best-effort: completed optimizer steps determine epoch/skip, but
    # async prompt/result buffers can still lose or replay a small tail.
    start_step = getattr(args, "resume_from_step", 0)
    resume_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    resume_skip = (start_step % steps_per_epoch) * controller.dispatch_batch_size if start_step > 0 else 0
    controller.submit_training_dataset(epoch=resume_epoch, skip=resume_skip)

    logger.info(
        f"Starting: num_steps={num_steps}, num_epochs={num_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"dispatch_batch_size={controller.dispatch_batch_size}"
    )

    enable_perf = getattr(args, "enable_perf_metrics", True)
    prefetch_batches = getattr(args, "prefetch_depth", 1)

    completed_steps = start_step
    current_epoch = completed_steps // steps_per_epoch + 1
    steps_in_current_epoch = completed_steps % steps_per_epoch
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch {current_epoch})")
    queued_batches = 0
    previous_dispatch_wait: float | None = None
    progress = tqdm(total=num_steps, desc="Running Inference", unit="step", initial=start_step)
    max_attempts_per_step = 100
    for step in range(start_step, num_steps):
        inference_manager_status = inference_manager.get_status()
        inference_prompt_buffer_size = inference_manager_status["prompt_buffer_size"]
        inference_pending_tasks = inference_manager_status["pending_tasks"]
        if _is_pipeline_idle():
            logger.info(f"Inference prompt buffer and controller sample pool drained at step {step}")
            break
        begin_next_epoch = steps_in_current_epoch >= steps_per_epoch and completed_steps < num_steps
        if begin_next_epoch:
            current_epoch += 1
            steps_in_current_epoch = 0
            logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
            controller.reload_dataset()
        deadline = time.monotonic() + 600
        while not controller.try_dispatch_batch():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Failed to dispatch batch after {max_attempts_per_step} attempts "
                    f"(epoch {current_epoch}, step {completed_steps})"
                )
            time.sleep(0.05)
        queued_batches += 1
        completed_steps += 1
        steps_in_current_epoch += 1
        progress.update(1)
        if queued_batches == prefetch_batches or step == num_steps - 1:
            samples_delete = controller.drain_queues(controller.train_queues)
            for sample in samples_delete:
                cleanup_mooncake_data(sample, mooncake_store)
            queued_batches = 0

    final_metric_step = int(completed_steps)
    final_metrics = {}
    final_metrics["train/step"] = final_metric_step
    final_metrics["inference/step"] = completed_steps
    if enable_perf and previous_dispatch_wait is not None:
        final_metrics["perf/dispatch_wait"] = previous_dispatch_wait
        step_time = final_metrics.get("perf/step_time", 0)
    _write_training_metrics(final_metrics, final_metric_step, completed_steps)

    progress.close()
    final_status = controller.get_full_status()
    logger.info(
        f"Training completed: {completed_steps} steps in {final_status['elapsed_seconds']:.1f}s | "
        f"avg inference={final_status['avg_inference_speed']:.1f} entries/s | "
        f"avg training={final_status['avg_training_speed']:.1f} entries/s"
    )


def run_training_loop(
    args,
    controller,
    inference_manager,
    mooncake_master,
    mooncake_store,
    dataset_size=None,
    eval_dataset_size=None,
):
    try:
        return training_loop(
            args,
            controller,
            inference_manager,
            mooncake_store,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
        )
    finally:
        _safe_training_cleanup(
            args=args,
            inference_manager=inference_manager,
            controller=controller,
            mooncake_master=mooncake_master,
            mooncake_store=mooncake_store,
        )
