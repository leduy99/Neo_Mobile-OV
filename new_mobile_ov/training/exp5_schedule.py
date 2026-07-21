from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Exp5Phase:
    name: str
    train_bridge: bool


@dataclass(frozen=True)
class Exp5Schedule:
    total_steps: int
    phase_a_steps: int
    phase_b_steps: int
    dit_warmup_steps: int
    final_cooldown_steps: int
    bridge_warmup_steps: int
    bridge_cooldown_steps: int
    flow_start_weight: float
    flow_peak_weight: float
    flow_final_weight: float

    def __post_init__(self) -> None:
        values = {
            "total_steps": self.total_steps,
            "phase_a_steps": self.phase_a_steps,
            "phase_b_steps": self.phase_b_steps,
            "dit_warmup_steps": self.dit_warmup_steps,
            "final_cooldown_steps": self.final_cooldown_steps,
            "bridge_warmup_steps": self.bridge_warmup_steps,
            "bridge_cooldown_steps": self.bridge_cooldown_steps,
        }
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Schedule steps must be non-negative: {values}")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive.")
        if self.phase_a_steps + self.phase_b_steps >= self.total_steps:
            raise ValueError("Phase C must contain at least one step.")
        if self.dit_warmup_steps > self.phase_a_steps:
            raise ValueError("DiT warmup must finish within Phase A.")
        if self.final_cooldown_steps > self.total_steps:
            raise ValueError("Final cooldown cannot exceed total_steps.")
        if self.bridge_warmup_steps + self.bridge_cooldown_steps > self.phase_b_steps:
            raise ValueError("Bridge warmup and cooldown must fit inside Phase B.")
        weights = [self.flow_start_weight, self.flow_peak_weight, self.flow_final_weight]
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"Flow weights must be non-negative: {weights}")

    @property
    def phase_b_end(self) -> int:
        return self.phase_a_steps + self.phase_b_steps

    @property
    def phase_c_steps(self) -> int:
        return self.total_steps - self.phase_b_end

    def phase(self, step: int) -> Exp5Phase:
        self._validate_step(step)
        if step <= self.phase_a_steps:
            return Exp5Phase("A_dit_warmup", train_bridge=False)
        if step <= self.phase_b_end:
            return Exp5Phase("B_joint_refinement", train_bridge=True)
        return Exp5Phase("C_dit_consolidation", train_bridge=False)

    def dit_lr_scale(self, step: int) -> float:
        self._validate_step(step)
        if self.dit_warmup_steps > 0 and step <= self.dit_warmup_steps:
            return step / float(self.dit_warmup_steps)
        cooldown_start = self.total_steps - self.final_cooldown_steps
        if self.final_cooldown_steps > 0 and step > cooldown_start:
            progress = (step - cooldown_start) / float(self.final_cooldown_steps)
            return 1.0 - 0.9 * progress
        return 1.0

    def bridge_lr_scale(self, step: int) -> float:
        phase = self.phase(step)
        if not phase.train_bridge:
            return 0.0
        relative_step = step - self.phase_a_steps
        if self.bridge_warmup_steps > 0 and relative_step <= self.bridge_warmup_steps:
            return relative_step / float(self.bridge_warmup_steps)
        cooldown_start = self.phase_b_steps - self.bridge_cooldown_steps
        if self.bridge_cooldown_steps > 0 and relative_step > cooldown_start:
            progress = (relative_step - cooldown_start) / float(self.bridge_cooldown_steps)
            return max(1.0 - progress, 0.0)
        return 1.0

    def flow_weight(self, step: int) -> float:
        self._validate_step(step)
        if self.dit_warmup_steps > 0 and step <= self.dit_warmup_steps:
            progress = step / float(self.dit_warmup_steps)
            return self.flow_start_weight + (self.flow_peak_weight - self.flow_start_weight) * progress
        cooldown_start = self.total_steps - self.final_cooldown_steps
        if self.final_cooldown_steps > 0 and step > cooldown_start:
            progress = (step - cooldown_start) / float(self.final_cooldown_steps)
            return self.flow_peak_weight + (self.flow_final_weight - self.flow_peak_weight) * progress
        return self.flow_peak_weight

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_steps": self.total_steps,
            "phase_a_steps": self.phase_a_steps,
            "phase_b_steps": self.phase_b_steps,
            "phase_b_end": self.phase_b_end,
            "phase_c_steps": self.phase_c_steps,
            "dit_warmup_steps": self.dit_warmup_steps,
            "final_cooldown_steps": self.final_cooldown_steps,
            "bridge_warmup_steps": self.bridge_warmup_steps,
            "bridge_cooldown_steps": self.bridge_cooldown_steps,
            "flow_start_weight": self.flow_start_weight,
            "flow_peak_weight": self.flow_peak_weight,
            "flow_final_weight": self.flow_final_weight,
        }

    def _validate_step(self, step: int) -> None:
        if not 1 <= step <= self.total_steps:
            raise ValueError(f"step must be in [1, {self.total_steps}], got {step}")
