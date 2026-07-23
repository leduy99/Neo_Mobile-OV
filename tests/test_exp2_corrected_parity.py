from __future__ import annotations

import shlex
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP2_CORRECTED = ROOT / "scripts/exp2_corrected_train_neodragon_joint_distill_1node8gpu.sbatch"
EXP3 = ROOT / "scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch"


def training_arguments(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    command = text.split("tools/train_neodragon_dit_bridge.py \\", 1)[1]
    command = command.split("${EXTRA_ARGS:-}", 1)[0]
    return shlex.split(command.replace("\\\n", " "))


def remove_option(arguments: list[str], option: str, *, takes_value: bool = True) -> list[str]:
    result = list(arguments)
    index = result.index(option)
    del result[index : index + (2 if takes_value else 1)]
    return result


class Exp2CorrectedParityTest(unittest.TestCase):
    def test_only_bridge_initialization_and_output_differ_from_exp3(self) -> None:
        exp2 = training_arguments(EXP2_CORRECTED)
        exp3 = training_arguments(EXP3)

        self.assertIn("--bridge-ckpt", exp2)
        self.assertNotIn("--bridge-ckpt", exp3)

        exp2 = remove_option(exp2, "--bridge-ckpt")
        exp2 = remove_option(exp2, "--output-dir")
        exp3 = remove_option(exp3, "--output-dir")
        self.assertEqual(exp2, exp3)


if __name__ == "__main__":
    unittest.main()
