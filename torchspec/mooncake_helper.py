import os
from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.train_entry import parse_config



def setup_mooncake(args):
    """
    Launch mooncake master and export mooncake related environment variables.
    """
    cfg  = MooncakeConfig.from_flat_args(args)     # __post_init__ computes host_buffer_size
    before = dict(os.environ)
    cfg.export_env()                                # writes exactly the right key set
    lines = ["#!/bin/bash"]
    for k, v in os.environ.items():
        if k.startswith(("MOONCAKE_", "MC_")) and before.get(k) != v:
            lines.append(f"export {k}={v}")
    final_string = "\n".join(lines)
    env_file = args.mooncake_env_file
    with open(env_file, "w") as f:
        f.write(final_string)

if __name__ == "__main__":
    args = parse_config()
    setup_mooncake(args)
