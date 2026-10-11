"""Helpers for finetuning WEAVER onto a new state/action space on one GPU.

Two problems this release's finetune path does not handle on its own:

1. **A different action space.**  The released checkpoint was trained with
   ``n_states = n_actions = 8`` (Franka joints + gripper).  Conditioning on a
   robot *and* a human hand needs 14, which changes the first layer of every
   state/action MLP and the state read-out.  ``load_state_dict`` is strict, so
   those tensors have to be filtered out and left at their fresh init.

2. **16 GB of VRAM.**  The world model is ~915 M parameters; a full AdamW
   finetune needs roughly 15 GB for parameters, gradients and moments alone.
   LoRA on the transformer blocks, with the (re-initialised) projections
   trained densely, brings the trainable set to ~35 M.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear layers inside each transformer block that LoRA adapts.
LORA_TARGETS = ("attn.qkv_proj", "attn.out_proj", "ffn.c_fc", "ffn.c_proj")

# Modules kept fully trainable: they either changed shape with the new action
# space or are cheap glue that benefits from adapting freely.
DENSE_TRAIN_PREFIXES = (
    "wm.inp_prj.",
    "wm.mem_inp_prj.",
    "wm.out_prj.",
    "wm.fusion_proj.",
    "wm.ln_f.",
    "wm.mask_token",
)


# --------------------------------------------------------------------------- #
# 1. shape-tolerant checkpoint loading
# --------------------------------------------------------------------------- #

def load_state_dict_shape_filtered(model: nn.Module, state_dict: dict,
                                   verbose: bool = True) -> dict:
    """Load every tensor whose shape matches; report the rest.

    Returns a summary dict with the skipped and missing key lists.  Skipped
    tensors keep the values ``WEAVER._init_weights`` gave them.
    """
    own = model.state_dict()
    to_load, mismatched = {}, []
    for k, v in state_dict.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            to_load[k] = v
        else:
            mismatched.append((k, tuple(v.shape), tuple(own[k].shape)))

    missing = [k for k in own if k not in state_dict]
    unexpected = [k for k in state_dict if k not in own]
    model.load_state_dict(to_load, strict=False)

    if verbose:
        print(f"[ckpt] loaded {len(to_load)}/{len(own)} tensors")
        if mismatched:
            print(f"[ckpt] {len(mismatched)} shape mismatches, left at fresh init:")
            for k, old, new in mismatched:
                print(f"         {k}: checkpoint {old} -> model {new}")
        if missing:
            print(f"[ckpt] {len(missing)} tensors absent from the checkpoint "
                  f"(e.g. {missing[:3]})")
        if unexpected:
            print(f"[ckpt] {len(unexpected)} checkpoint tensors unused "
                  f"(e.g. {unexpected[:3]})")
    return {"loaded": list(to_load), "mismatched": mismatched,
            "missing": missing, "unexpected": unexpected}


# --------------------------------------------------------------------------- #
# 2. LoRA
# --------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """``nn.Linear`` with a frozen base and a trainable low-rank update.

    Written out rather than pulled from peft because peft rewrites parameter
    paths (``base_model.model....``), which would break ``save_checkpoint`` /
    ``generate_views`` round-tripping for this model.  Here the base keeps its
    own name under ``.base`` and only ``lora_A`` / ``lora_B`` are added.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # Match the layer being wrapped, otherwise the adapters land on the CPU
        # while the model is already on the GPU.
        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **kw))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **kw))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.lora_A.shape[0]}, scaling={self.scaling:.3g}"


def _set_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, value)


def apply_lora(model: nn.Module, rank: int = 32, alpha: float = 64.0,
               dropout: float = 0.0, verbose: bool = True) -> int:
    """Wrap the transformer blocks' linears in LoRA. Returns how many were wrapped.

    Must run *after* the pretrained checkpoint is loaded when starting a fresh
    finetune (wrapping renames ``...qkv_proj.weight`` to ``...qkv_proj.base.weight``),
    and *before* loading when resuming a finetune checkpoint that already has
    the adapters.
    """
    targets = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and "self_attn_blocks" in name
        and name.endswith(LORA_TARGETS)
    ]
    for name in targets:
        base = model.get_submodule(name)
        _set_submodule(model, name, LoRALinear(base, rank, alpha, dropout))
    if verbose:
        print(f"[lora] wrapped {len(targets)} linear layers (rank={rank}, alpha={alpha})")
    return len(targets)


def freeze_for_lora(model: nn.Module, verbose: bool = True) -> tuple[int, int]:
    """Freeze everything except the LoRA adapters and the dense-train modules.

    The reward and critic heads are frozen too: their losses are switched off
    for this finetune (no success labels exist for the play data), so training
    them would only burn optimiser state.
    """
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        keep = (
            ".lora_A" in name or ".lora_B" in name
            or name.startswith(DENSE_TRAIN_PREFIXES)
        )
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
        else:
            frozen += p.numel()

    if verbose:
        print(f"[lora] trainable {trainable / 1e6:.1f}M / "
              f"{(trainable + frozen) / 1e6:.1f}M parameters "
              f"({100 * trainable / (trainable + frozen):.2f}%)")
        groups = sorted({
            n.rsplit(".", 1)[0] for n, p in model.named_parameters()
            if p.requires_grad and ".lora_" not in n
        })
        print(f"[lora] densely trained modules: {groups}")
    return trainable, frozen


def rebuild_ema(model: nn.Module, beta: float = 0.9999) -> None:
    """Re-create ``model.ema`` over the current trainable parameter set.

    ``WEAVER.__init__`` snapshots ``requires_grad`` parameters when it builds the
    EMA, which at that point is the whole 915 M world model — 3.6 GB of shadow
    weights that are now both wrong (the set changed) and unaffordable.
    """
    from .utils.tools import EMA
    device = next(model.parameters()).device
    model.ema = EMA(model, beta=beta)
    model.ema.to(device)
    n = sum(p.numel() for p in model.ema.params)
    print(f"[ema] rebuilt over {n / 1e6:.1f}M trainable parameters")


def trainable_parameter_names(model: nn.Module) -> Iterable[str]:
    return (n for n, p in model.named_parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# 3. loading a finetune back
# --------------------------------------------------------------------------- #

def load_finetuned(model: nn.Module, checkpoint_dir: str, cfg,
                   checkpoint_name: str = "checkpoint.pt",
                   device: str = "cpu", verbose: bool = True) -> dict:
    """Restore a finetuned model, LoRA adapters and all.

    A LoRA finetune saves only its trainable tensors, so the frozen remainder has
    to come from the pretrained checkpoint first — and *before* the adapters are
    wrapped on, because wrapping renames ``...qkv_proj.weight`` to
    ``...qkv_proj.base.weight``. Order therefore matters:

        build model -> load pretrained base -> apply LoRA -> load the adapters

    Works unchanged on a full (non-LoRA) checkpoint, where the first two steps
    are skipped.  Returns the loaded checkpoint dict.
    """
    import os

    from .utils.tools import load_checkpoint

    ckpt = load_checkpoint(checkpoint_dir, "cpu", checkpoint_name=checkpoint_name)
    partial = bool(ckpt.get("trainable_only", False))
    lora = getattr(cfg, "lora", None)
    rank = getattr(lora, "rank", 0) if lora is not None else 0

    if partial:
        pretrained = getattr(cfg, "pretrained_dir", None)
        if not pretrained or not os.path.exists(os.path.join(pretrained, "checkpoint.pt")):
            raise FileNotFoundError(
                f"{checkpoint_name} holds only the trainable tensors, so the frozen "
                f"weights must come from the pretrained checkpoint, but "
                f"cfg.pretrained_dir={pretrained!r} does not hold one. Pass the "
                f"right path via --overrides pretrained_dir=/path/to/chkpts."
            )
        if verbose:
            print(f"[ckpt] partial checkpoint; taking the frozen weights from {pretrained}")
        # Onto the CPU: the pretrained checkpoint carries model + EMA + optimiser
        # (~8 GB) and will not fit beside an already-built model on a 16 GB card.
        # load_state_dict copies to the parameters' own device anyway.
        base = load_checkpoint(pretrained, "cpu")
        load_state_dict_shape_filtered(model, base["model"], verbose=verbose)
        del base

    if rank > 0:
        apply_lora(model, rank, lora.alpha, 0.0, verbose=verbose)

    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in ckpt["model"].items()}
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=not partial)
    if verbose and partial:
        print(f"[ckpt] applied {len(state)} finetuned tensors "
              f"({len(unexpected)} unexpected)")

    if partial:
        # model.ema was built over every parameter when the model was
        # constructed, but the saved EMA covers only the trained ones. Repeat
        # the training-time freeze and rebuild so the two line up, otherwise
        # ema.load_state_dict / apply_to walk mismatched tensors.
        freeze_for_lora(model, verbose=False)
        rebuild_ema(model)
    return ckpt


# --------------------------------------------------------------------------- #
# 4. optimiser groups
# --------------------------------------------------------------------------- #

def configure_optimizers_split(model: nn.Module, weight_decay: float,
                               learning_rate: float, betas: tuple,
                               projection_lr_mult: float = 1.0,
                               verbose: bool = True):
    """AdamW with the freshly-initialised projections on their own learning rate.

    The LoRA adapters refine weights that are already trained; the state/action
    projections start from random because the action space changed, and they
    have much further to travel. Running both at one rate lets the model settle
    into ignoring whichever action channels it can predict from the images
    instead — which is exactly what happened to the hand on the first run.

    Each group carries ``lr_scale`` so the schedule can keep the ratio while it
    warms up and decays.
    """
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_projection = ".lora_A" not in name and ".lora_B" not in name
        key = (is_projection, p.dim() >= 2)
        groups.setdefault(key, []).append(p)

    optim_groups = []
    for (is_projection, decay), params in sorted(groups.items()):
        scale = projection_lr_mult if is_projection else 1.0
        optim_groups.append({
            "params": params,
            "weight_decay": weight_decay if decay else 0.0,
            "lr": learning_rate * scale,
            "lr_scale": scale,
            "group_name": ("projection" if is_projection else "lora")
                          + ("/decay" if decay else "/nodecay"),
        })

    if verbose:
        for g in optim_groups:
            n = sum(p.numel() for p in g["params"])
            print(f"[optim] {g['group_name']:<18} {n / 1e6:7.2f}M params  "
                  f"lr x{g['lr_scale']:g}")

    import inspect
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                             **({"fused": True} if fused else {}))
