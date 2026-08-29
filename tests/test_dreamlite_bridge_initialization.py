from pathlib import Path

import pytest
import torch

from tools.train_dreamlite_image_bridge import initialize_bridge_from_checkpoint


class FakeBridge:
    def __init__(self) -> None:
        self.loaded = None

    def load_trainable_state_dict(self, state) -> None:
        self.loaded = state


def test_initialization_loads_weights_but_reports_reset_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v11.pt"
    state = {"projection.weight": torch.ones(2, 2)}
    torch.save(
        {
            "step": 160000,
            "bridge": state,
            "optimizer": {"state": {1: "must not be loaded"}},
            "architecture": "MobileOVDreamLiteCompactBridgeV11",
            "config": {"training_version": "v11_balanced"},
        },
        checkpoint,
    )

    bridge = FakeBridge()
    metadata = initialize_bridge_from_checkpoint(bridge, checkpoint)

    assert bridge.loaded is state or torch.equal(
        bridge.loaded["projection.weight"], state["projection.weight"]
    )
    assert metadata == {
        "checkpoint": str(checkpoint),
        "source_step": 160000,
        "source_training_version": "v11_balanced",
        "source_architecture": "MobileOVDreamLiteCompactBridgeV11",
        "optimizer_reset": True,
        "step_reset": True,
    }


def test_initialization_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing bridge initialization checkpoint"):
        initialize_bridge_from_checkpoint(FakeBridge(), tmp_path / "missing.pt")


def test_initialization_accepts_raw_bridge_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "raw.pt"
    torch.save({"projection.weight": torch.zeros(1)}, checkpoint)
    bridge = FakeBridge()

    metadata = initialize_bridge_from_checkpoint(bridge, checkpoint)

    assert torch.equal(bridge.loaded["projection.weight"], torch.zeros(1))
    assert metadata["source_step"] == -1
    assert metadata["optimizer_reset"] is True
