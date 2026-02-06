"""
Shock Profiles for Clock 3 Validation

Named stress regimes that create differentiated failure modes across layers.
The key insight: uniform brutality causes uniform collapse; structured shocks
create separable failure modes where priors can bite.

Usage:
    from experiments.shock_profiles import DIFFERENTIATED_8L, ShockProfile

    profile = DIFFERENTIATED_8L
    config = profile.to_train_config()
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ShockProfile:
    """
    A named stress regime for MoE training.

    Design principle: the profile should create enough pressure to reveal
    layer-specific fragility, but not so much that everything collapses
    uniformly.
    """
    name: str
    description: str

    # Model architecture
    n_layer: int = 4
    n_exp: int = 4
    top_k: int = 1
    stride: int = 1  # 1 = every layer is MoE
    n_embd: int = 128
    n_head: int = 4
    block_size: int = 128  # Sequence length

    # Capacity pressure
    train_capacity: float = 1.0
    eval_capacity: float = 2.0

    # Load balancing help (lower = less help = more natural collapse)
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001

    # Controller sensitivity
    chrono_neff_threshold_ratio: float = 0.85
    chrono_top2_warning: float = 0.60
    chrono_lens_rank: int = 4

    # Training dynamics
    learning_rate: float = 3e-4
    warmup_iters: int = 50
    dropout: float = 0.1

    # Timing
    eval_interval: int = 25

    def to_train_config(self, **overrides) -> Dict[str, Any]:
        """Convert to training config dict, with optional overrides."""
        config = {
            'n_layer': self.n_layer,
            'n_exp': self.n_exp,
            'top_k': self.top_k,
            'stride': self.stride,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'block_size': self.block_size,
            'train_capacity': self.train_capacity,
            'eval_capacity': self.eval_capacity,
            'use_aux_loss': self.aux_loss_weight > 0,
            'aux_loss_weight': self.aux_loss_weight,
            'use_router_z_loss': self.router_z_loss_weight > 0,
            'router_z_loss_weight': self.router_z_loss_weight,
            'chrono_neff_threshold_ratio': self.chrono_neff_threshold_ratio,
            'chrono_top2_warning': self.chrono_top2_warning,
            'chrono_lens_rank': self.chrono_lens_rank,
            'learning_rate': self.learning_rate,
            'warmup_iters': self.warmup_iters,
            'dropout': self.dropout,
            'eval_interval': self.eval_interval,
        }
        config.update(overrides)
        return config

    def to_config_str(self, **overrides) -> str:
        """Generate Python config file content."""
        config = self.to_train_config(**overrides)
        lines = [f"# ShockProfile: {self.name}", f"# {self.description}", ""]
        for k, v in config.items():
            if isinstance(v, str):
                lines.append(f"{k} = '{v}'")
            elif isinstance(v, bool):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f"{k} = {v}")
        return "\n".join(lines)


# =============================================================================
# Named Profiles
# =============================================================================

GENTLE = ShockProfile(
    name="GENTLE",
    description="Minimal stress baseline. Used for sanity checks.",
    n_layer=4,
    n_exp=4,
    top_k=2,  # More forgiving
    train_capacity=1.25,
    eval_capacity=2.0,
    aux_loss_weight=0.01,
    router_z_loss_weight=0.001,
    chrono_neff_threshold_ratio=0.85,
    warmup_iters=100,
)

MODERATE = ShockProfile(
    name="MODERATE",
    description="Moderate pressure. Creates some differentiation.",
    n_layer=4,
    n_exp=4,
    top_k=1,
    train_capacity=1.0,
    eval_capacity=2.0,
    aux_loss_weight=0.01,
    router_z_loss_weight=0.001,
    chrono_neff_threshold_ratio=0.85,
    warmup_iters=50,
)

HARSH = ShockProfile(
    name="HARSH",
    description="Strong pressure. Tests controller limits.",
    n_layer=4,
    n_exp=4,
    top_k=1,
    train_capacity=0.8,
    eval_capacity=1.5,
    aux_loss_weight=0.005,
    router_z_loss_weight=0.0005,
    chrono_neff_threshold_ratio=0.90,
    chrono_top2_warning=0.55,
    warmup_iters=20,
    dropout=0.15,
)

DIFFERENTIATED_8L = ShockProfile(
    name="DIFFERENTIATED_8L",
    description="8 MoE layers with moderate stress. Validated r=0.995 targeting.",
    n_layer=8,
    n_exp=4,
    top_k=1,
    stride=1,  # All 8 layers are MoE
    train_capacity=0.8,
    eval_capacity=1.5,
    aux_loss_weight=0.005,
    router_z_loss_weight=0.0005,
    chrono_neff_threshold_ratio=0.90,
    chrono_top2_warning=0.55,
    learning_rate=5e-4,
    warmup_iters=20,
    dropout=0.15,
    eval_interval=15,
)

NANOMOE_PLUS = ShockProfile(
    name="NANOMOE_PLUS",
    description="16 experts, top-2. NOTE: Too stable - controller rarely triggers.",

    # Architecture - more experts = more routing ambiguity
    n_layer=12,
    n_exp=16,           # Was 4, now 16 for genuine competition
    top_k=2,            # Top-2 routing = more interference
    stride=1,           # All layers are MoE
    n_embd=256,         # Richer representations
    n_head=4,
    block_size=256,     # Longer temporal dependencies

    # Capacity pressure - force token drops
    train_capacity=0.7,  # Tight: not all tokens get preferred expert
    eval_capacity=1.2,   # Slightly relaxed for eval

    # Weak load balancing - let natural collapse happen
    aux_loss_weight=0.003,
    router_z_loss_weight=0.0003,

    # Sensitive controller
    chrono_neff_threshold_ratio=0.85,
    chrono_top2_warning=0.50,
    chrono_lens_rank=8,

    # Training dynamics
    learning_rate=3e-4,
    warmup_iters=30,
    dropout=0.1,
    eval_interval=20,
)

# The profile that achieves meaningful targeting correlation
# without uniform collapse
VALIDATED = DIFFERENTIATED_8L


# =============================================================================
# Profile Registry
# =============================================================================

PROFILES = {
    'gentle': GENTLE,
    'moderate': MODERATE,
    'harsh': HARSH,
    'differentiated_8l': DIFFERENTIATED_8L,
    'nanomoe_plus': NANOMOE_PLUS,
    'validated': VALIDATED,
}


def get_profile(name: str) -> ShockProfile:
    """Get a profile by name (case-insensitive)."""
    key = name.lower()
    if key not in PROFILES:
        available = ', '.join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[key]
