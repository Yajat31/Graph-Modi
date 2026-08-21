from pathlib import Path

from graph_modi.config import load_config


def test_data_dir_defaults_under_output_dir(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 1\noutput_dir: outputs/example\n", encoding="utf-8")
    config = load_config(path)
    assert config.data_dir == Path("outputs/example/data")


def test_data_dir_can_point_at_shared_datasets(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "seed: 1\noutput_dir: outputs/example\ndata_dir: datasets/example\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.data_dir == Path("datasets/example")
    assert config.output_dir == Path("outputs/example")
