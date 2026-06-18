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


class LoRAQKVLinear(nn.Module):
    """LoRA adapter for fused qkv projections, updating q and v only."""

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
        if base_layer.out_features % 3 != 0:
            raise ValueError(
                f"fused qkv out_features must be divisible by 3, got {base_layer.out_features}"
            )

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.qkv_dim = base_layer.out_features // 3

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B_q = nn.Linear(r, self.qkv_dim, bias=False)
        self.lora_B_v = nn.Linear(r, self.qkv_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        hidden = self.lora_A(self.dropout(x))
        delta_q = self.lora_B_q(hidden)
        delta_v = self.lora_B_v(hidden)
        delta_k = torch.zeros_like(delta_q)
        delta = torch.cat((delta_q, delta_k, delta_v), dim=-1)
        return base + delta * self.scaling


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 4
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    target_prefixes: Tuple[str, ...] = ()
    freeze_base_model: bool = True


def _get_parent_module(model: nn.Module, module_name: str) -> nn.Module:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def _matches_target(module_name: str, target_modules: Iterable[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def _matches_prefix(module_name: str, target_prefixes: Iterable[str]) -> bool:
    prefixes = tuple(target_prefixes)
    if len(prefixes) == 0:
        return True
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(),
) -> int:
    if config.freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

    replacements = []
    for module_name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and _matches_prefix(module_name, config.target_prefixes)
            and _matches_target(module_name, config.target_modules)
        ):
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


def apply_qv_lora_to_fused_qkv(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(target_modules=("qkv",)),
) -> int:
    if config.freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

    replacements = []
    for module_name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and _matches_prefix(module_name, config.target_prefixes)
            and _matches_target(module_name, config.target_modules)
        ):
            replacements.append((module_name, module))

    for module_name, module in replacements:
        parent = _get_parent_module(model, module_name)
        child_name = module_name.rsplit(".", 1)[-1]
        setattr(
            parent,
            child_name,
            LoRAQKVLinear(
                base_layer=module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )

    return len(replacements)
