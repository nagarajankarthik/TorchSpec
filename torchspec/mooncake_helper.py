"""Write the Mooncake environment file consumed by the vLLM connector.

``launch.sh`` runs this before starting ``mooncake_master`` and vLLM.  It
resolves the same config the training driver will use, renders it into the
``MOONCAKE_*`` / ``MC_*`` environment variables that
``MooncakeConfig.from_env()`` reads inside the vLLM worker, and writes them
to a shell snippet for ``launch.sh`` to source.

It does NOT launch the Mooncake master -- that is started directly from
``launch.sh`` so it outlives this short-lived process.
"""

import os

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.train_entry import parse_config


def setup_mooncake(args):
    """Render the resolved Mooncake config into ``args.mooncake_env_file``."""
    cfg = MooncakeConfig.from_flat_args(args)  # __post_init__ computes host_buffer_size
    before = dict(os.environ)
    cfg.export_env()  # writes exactly the right key set
    lines = ["#!/bin/bash"]
    for k, v in os.environ.items():
        if k.startswith(("MOONCAKE_", "MC_")) and before.get(k) != v:
            lines.append(f"export {k}={v}")
    final_string = "\n".join(lines)
    env_file = args.mooncake_env_file
    with open(env_file, "w") as f:
        f.write(final_string)


if __name__ == "__main__":
    # save_snapshot=False: train_entry writes output_dir/config.yaml from the
    # same config moments later, so writing it here too is redundant.
    args = parse_config(save_snapshot=False)
    setup_mooncake(args)
