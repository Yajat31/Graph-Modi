import json

from graph_modi.models.tea_glm import (
    TEA_CHECKPOINT_FORMAT,
    load_checkpoint_metadata,
)
from graph_modi.training import run_mock_training


def test_download_free_training_checkpoint_has_reproducibility_metadata(
    tmp_path,
) -> None:
    metadata_path = run_mock_training(tmp_path, seed=123, split_name="train")
    metadata = load_checkpoint_metadata(metadata_path)
    assert metadata["format"] == TEA_CHECKPOINT_FORMAT
    assert metadata["mock"] is True
    assert metadata["training_config"]["seed"] == 123
    assert metadata["data_sha256"]

    result = json.loads((tmp_path / "training_result.json").read_text())
    assert result["final_checkpoint"].endswith("checkpoint-mock")
