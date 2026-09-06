"""Write the endpoint environment file shared by every process in the run.

``launch.sh`` runs this before starting ``mooncake_master``, Redis and vLLM. It
resolves the same config the training driver will use and renders it into a
shell snippet for ``launch.sh`` to source:

* ``MOONCAKE_*`` / ``MC_*`` -- read by ``MooncakeConfig.from_env()`` inside the
  vLLM worker.
* ``REDIS_TRAIN_STREAM`` / ``REDIS_EVAL_STREAM`` -- the stream names the
  controller publishes to, emitted here so ``launch.sh`` does not hardcode a
  second copy that can drift from the config. A drift would be silent: the
  producer would write to one stream while consumers blocked on another.

The file lands in ``$LOG_DIR`` on shared storage, so it doubles as the
discovery contract for consumers running in a different scheduler job.

It does NOT launch the Mooncake master -- that is started directly from
``launch.sh`` so it outlives this short-lived process.
"""

import os

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.train_entry import parse_config


def setup_mooncake(args):
    """Render the resolved endpoint config into ``args.mooncake_env_file``."""
    cfg = MooncakeConfig.from_flat_args(args)  # __post_init__ computes host_buffer_size
    before = dict(os.environ)
    cfg.export_env()  # writes exactly the right key set
    lines = ["#!/bin/bash"]
    for k, v in os.environ.items():
        if k.startswith(("MOONCAKE_", "MC_")) and before.get(k) != v:
            lines.append(f"export {k}={v}")

    # Single source of truth for the stream names: the controller reads these
    # from the config, so consumers must get them from the config too.
    lines.append(f"export REDIS_TRAIN_STREAM={getattr(args, 'redis_train_stream', 'train_samples')}")
    lines.append(f"export REDIS_EVAL_STREAM={getattr(args, 'redis_eval_stream', 'eval_samples')}")

    final_string = "\n".join(lines)
    env_file = args.mooncake_env_file
    with open(env_file, "w") as f:
        f.write(final_string)


if __name__ == "__main__":
    # save_snapshot=False: train_entry writes output_dir/config.yaml from the
    # same config moments later, so writing it here too is redundant.
    args = parse_config(save_snapshot=False)
    setup_mooncake(args)
