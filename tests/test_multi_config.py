"""Model config + cluster config merging, as used by custom_scripts/."""

import sys

import yaml
from omegaconf import OmegaConf

from torchspec.config.train_config import load_config, save_config


def _write(path, data):
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_later_file_overrides_only_the_keys_it_sets(tmp_path):
    model = _write(tmp_path / "model.yaml", {
        "mooncake": {"protocol": "rdma", "device_name": "mlx5_0", "num_aux_layers": 3},
    })
    cluster = _write(tmp_path / "cluster.yaml", {"mooncake": {"protocol": "tcp", "device_name": ""}})

    config = load_config(config_path=[model, cluster])
    assert config.mooncake.protocol == "tcp"
    assert config.mooncake.device_name == ""
    assert config.mooncake.num_aux_layers == 3  # untouched by the cluster file


def test_cli_overrides_beat_every_file(tmp_path):
    a = _write(tmp_path / "a.yaml", {"mooncake": {"protocol": "rdma"}})
    b = _write(tmp_path / "b.yaml", {"mooncake": {"protocol": "tcp"}})
    config = load_config(config_path=[a, b], cli_args=["mooncake.protocol=rdma"])
    assert config.mooncake.protocol == "rdma"


def test_single_path_string_still_works(tmp_path):
    a = _write(tmp_path / "a.yaml", {"mooncake": {"protocol": "rdma"}})
    assert load_config(config_path=a).mooncake.protocol == "rdma"


def test_base_dir_interpolation_resolves_and_is_saved_resolved(tmp_path, monkeypatch):
    data = tmp_path / "data_bank" / "ds"
    data.mkdir(parents=True)
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    (tmp_path / "cfg").mkdir()
    cfg = _write(tmp_path / "cfg" / "m.yaml", {
        "dataset": {"train_data_path": "${oc.env:BASE_DIR}/data_bank/ds"},
    })

    config = load_config(config_path=cfg)
    # Not mistaken for a path relative to the config file's directory.
    assert config.dataset.train_data_path == str(data)

    snapshot = tmp_path / "snapshot.yaml"
    save_config(config, str(snapshot))
    assert OmegaConf.load(snapshot).dataset.train_data_path == str(data)


def test_parse_config_accepts_repeated_config_and_dotlist(tmp_path, monkeypatch):
    from torchspec.train_entry import parse_config

    model = _write(tmp_path / "model.yaml", {"mooncake": {"protocol": "rdma", "num_aux_layers": 3}})
    cluster = _write(tmp_path / "cluster.yaml", {"mooncake": {"protocol": "tcp"}})
    monkeypatch.setattr(sys, "argv", [
        "train_entry", "--config", model, "--config", cluster, "mooncake.local_hostname=10.0.0.1",
    ])
    args = parse_config(save_snapshot=False)
    assert args.mooncake_protocol == "tcp"
    assert args.mooncake_num_aux_layers == 3
    assert args.mooncake_local_hostname == "10.0.0.1"
