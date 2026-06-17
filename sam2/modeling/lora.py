from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        alpha: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 4
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    freeze_base_model: bool = True


def _get_parent_module(model: nn.Module, module_name: str) -> nn.Module:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def _matches_target(module_name: str, target_modules: Iterable[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(),
) -> int:
    if config.freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

    replacements = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _matches_target(module_name, config.target_modules):
            replacements.append((module_name, module))

    for module_name, module in replacements:
        parent = _get_parent_module(model, module_name)
        child_name = module_name.rsplit(".", 1)[-1]
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer=module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )

    return len(replacements)
