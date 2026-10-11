"""Finetune WEAVER from a pretrained checkpoint.

This script intentionally keeps only the WEAVER finetuning path: load a
pretrained checkpoint, optionally resume an existing finetune checkpoint, train on
``cfg.dataset``, and save to ``logs/chkpts_<suffix>``. Architecture fields are
inherited from the pretrained run so the finetune command only needs data and
training overrides.
"""

import argparse
import copy
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
import yaml
from einops import rearrange
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .finetune_utils import (
    apply_lora, configure_optimizers_split, freeze_for_lora,
    load_state_dict_shape_filtered, rebuild_ema,
)
from .utils.config import dict_to_namespace, merge_dicts, parse_config, update_config
from .wm.encoders import get_encoder, get_task_encoder
from .wm.model import WEAVER
from .wm.nets import STAttentionBlock
from .datasets import create_dataset
from .utils.tools import cycle, get_lr, load_checkpoint, move_tensors_to_device, save_checkpoint
from .utils.eval import evaluate_with_metrics_ddp


torch.set_float32_matmul_precision("high")


PRETRAINED_INHERIT_KEYS = [
    "model",
    "im_encoder",
    "n_history",
    "horizon",
    "n_memory_frames",
    "t_memory",
    "eval_horizon",
    "eval_video_frames",
    "eval_bootstrap",
    "inference",
]


def load_pretrained_config(pretrained_dir: str) -> dict | None:
    config_path = Path(pretrained_dir) / "config.yaml"
    if not config_path.exists():
        print(f"WARNING: no pretrained config found at {config_path}")
        return None
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def inherit_pretrained_config(cfg_dict: dict, pretrained_cfg: dict | None) -> dict:
    if pretrained_cfg is None:
        return cfg_dict

    cfg_dict = copy.deepcopy(cfg_dict)
    for key in PRETRAINED_INHERIT_KEYS:
        if key in pretrained_cfg:
            if isinstance(cfg_dict.get(key), dict) and isinstance(pretrained_cfg[key], dict):
                cfg_dict[key] = merge_dicts(copy.deepcopy(cfg_dict[key]), copy.deepcopy(pretrained_cfg[key]))
            else:
                cfg_dict[key] = pretrained_cfg[key]

    pretrained_dataset = pretrained_cfg.get("dataset", {})
    cfg_dict.setdefault("dataset", {})
    # img_keys / image_size must match the pretrained tokenisation. n_states and
    # n_actions must NOT be inherited: finetuning onto a different action space
    # (e.g. robot EEF + human hand) is the whole point of changing them, and the
    # affected projections are handled by load_state_dict_shape_filtered.
    for key in ["img_keys", "image_size"]:
        if key in pretrained_dataset:
            cfg_dict["dataset"][key] = pretrained_dataset[key]

    return cfg_dict


def apply_finetune_config(cfg_dict: dict) -> dict:
    cfg_dict = copy.deepcopy(cfg_dict)
    finetune_cfg = cfg_dict.get("finetune_cfg", {})
    if finetune_cfg:
        cfg_dict.setdefault("training", {})
        cfg_dict["training"].update(finetune_cfg)
        print(f"Applied finetune_cfg to training: {finetune_cfg}")
    return cfg_dict


def setup_ddp() -> tuple[bool, int, int, int, bool]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(f"cuda:{local_rank}")
        return True, rank, local_rank, world_size, rank == 0
    return False, 0, 0, 1, True


def image_keys_from_cfg(cfg) -> list[str]:
    if cfg.dataset.sample_aux_from_left_right:
        return ["aux", "wrist"]
    return list(cfg.dataset.img_keys)


def create_loaders(cfg, ddp: bool, batch_size: int):
    train_loader = create_dataset(
        cfg.dataset,
        horizon=cfg.horizon,
        ddp=ddp,
        batch_size=batch_size,
        n_workers=cfg.dataset.n_workers,
        split="train",
        return_video_frames=False,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    valid_loader = create_dataset(
        cfg.dataset,
        horizon=max(2 * cfg.horizon, 24),
        ddp=ddp,
        batch_size=4,
        n_workers=1,
        max_trajectories=200,
        split="val",
        return_video_frames=True,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    valid_video_loader = create_dataset(
        cfg.dataset,
        horizon=cfg.eval_video_frames,
        ddp=ddp,
        batch_size=4,
        n_workers=1,
        max_trajectories=None,
        split="val",
        return_video_frames=True,
        im_encoder_name=cfg.im_encoder.name,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        n_history=cfg.n_history,
    )
    return train_loader, valid_loader, valid_video_loader


def build_weaver(cfg):
    im_encoder, train_decoder = get_encoder(
        config=cfg.im_encoder,
        image_size=cfg.dataset.image_size,
        device="cuda",
    )
    task_encoder = get_task_encoder(config=None, device="cuda")

    image_size = cfg.dataset.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    model = WEAVER(
        img_keys=cfg.dataset.img_keys,
        im_encoder=im_encoder,
        train_decoder=train_decoder,
        task_encoder=task_encoder,
        n_history=cfg.n_history,
        n_horizon=cfg.horizon,
        config=cfg.model,
        use_precomputed_features=cfg.use_precomputed_features,
        n_states=cfg.dataset.n_states,
        n_actions=cfg.dataset.n_actions,
        image_size=image_size,
        device="cuda",
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        inference_config=cfg.inference,
    ).to("cuda")
    model.ema.to("cuda")
    return model


def enable_activation_checkpointing(model, master_process: bool):
    if not master_process:
        return
    n_ckpt = sum(isinstance(module, STAttentionBlock) for module in model.modules())
    print(f"Activation checkpointing enabled on {n_ckpt} STAttentionBlock layers")


def save_validation_videos(raw_model, valid_video_iter, cfg, img_keys, vid_dir, step,
                           n_batches: int = 1, use_wandb: bool = False):
    """Write a few GT-vs-prediction clips.

    Each clip is an autoregressive rollout over ``eval_video_frames`` and costs
    about a minute, so the batch count is kept low: this is a sanity glance
    between checkpoints, not the evaluation (that is scripts/eval_local.sh).
    """
    raw_model.eval()
    n_vids = 4
    sample_idx = 0
    for _ in range(n_batches):
        val_data = next(valid_video_iter)
        val_data = move_tensors_to_device(val_data, device="cuda")
        val_memory = {k: v[:n_vids] for k, v in val_data["memory"].items()} if "memory" in val_data else None

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, decoded_obs_pred = raw_model.generate_videos_full(
                obs={k: v[:n_vids] for k, v in val_data["obs"].items()},
                actions=val_data["actions"][:n_vids],
                instructions={k: v[:n_vids] for k, v in val_data["task"].items()},
                horizon=cfg.eval_horizon,
                memory=val_memory,
                bootstrap=cfg.eval_bootstrap,
            )

        t_pred = decoded_obs_pred[img_keys[0]].shape[1]
        for i in range(n_vids):
            gt_views = [
                rearrange(val_data["obs"][key][i:i + 1, :t_pred].float().cpu(), "b t c h w -> t h (b w) c")
                for key in img_keys
            ]
            pred_views = [
                rearrange(decoded_obs_pred[key][i:i + 1].float().cpu(), "b t c h w -> t h (b w) c")
                for key in img_keys
            ]
            video = np.concatenate([np.concatenate(gt_views, axis=2), np.concatenate(pred_views, axis=2)], axis=1)
            frames = (video * 255).astype(np.uint8)
            path = os.path.join(vid_dir, f"wm_valid_inference_step{step}_sample{sample_idx}.mp4")
            imageio.mimwrite(path, frames, fps=5)
            if use_wandb:
                import wandb
                # Hand wandb the file just written, not the raw array: passing an
                # array makes it re-encode through moviepy, an extra dependency
                # for a video that already exists on disk.
                wandb.log({f"valid/rollout_{sample_idx}": wandb.Video(path, format="mp4")},
                          step=step)
            sample_idx += 1

        del val_data, val_memory, decoded_obs_pred
        torch.cuda.empty_cache()
    raw_model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--mode", type=str, default="defaults", choices=["defaults", "debug"])
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--finetune_suffix", type=str, default="finetune")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank for the transformer blocks; 0 = full finetune. "
                             "Needed to fit a ~915M-parameter finetune on a single 16GB GPU.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha (default: 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--projection_lr_mult", type=float, default=1.0,
                        help="learning-rate multiplier for the state/action "
                             "projections. They are re-initialised whenever the "
                             "action space changes, so they start much further "
                             "from their target than the LoRA adapters and a "
                             "shared rate lets the model settle into ignoring "
                             "the action channels it can guess from the images.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2.0 * args.lora_rank

    cfg, cfg_dict = parse_config(args)
    cfg_dict = inherit_pretrained_config(cfg_dict, load_pretrained_config(args.pretrained_dir))
    cfg_dict = apply_finetune_config(cfg_dict)
    cfg_dict = update_config(cfg_dict, dict(item.split("=", 1) for item in args.overrides))
    # Recorded in the saved config so inference (generate_views, the interactive
    # demo) can rebuild the same architecture before loading the weights.
    cfg_dict["lora"] = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "projection_lr_mult": args.projection_lr_mult,
    }
    # A LoRA finetune stores only its trainable tensors, so anything loading it
    # later needs to know where the frozen remainder came from.
    cfg_dict["pretrained_dir"] = os.path.abspath(args.pretrained_dir)
    cfg = dict_to_namespace(cfg_dict)

    print("Arguments:", cfg)
    log_dir = os.path.join(cfg.scratch_dir, "logs")
    chkpt_save_dir = os.path.join(log_dir, f"chkpts_{args.finetune_suffix}")
    vid_dir = os.path.join(log_dir, "videos")
    os.makedirs(chkpt_save_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    ddp, _, local_rank, world_size, master_process = setup_ddp()
    img_keys = image_keys_from_cfg(cfg)
    train_loader, valid_loader, valid_video_loader = create_loaders(cfg, ddp, cfg.training.batch_size)

    model = build_weaver(cfg)
    finetune_ckpt = os.path.join(chkpt_save_dir, "checkpoint.pt")
    resume_finetune = os.path.exists(finetune_ckpt)
    load_dir = chkpt_save_dir if resume_finetune else args.pretrained_dir
    if not os.path.exists(os.path.join(load_dir, "checkpoint.pt")):
        raise FileNotFoundError(f"No checkpoint.pt found in {load_dir}")

    # Onto the CPU, not the GPU: checkpoint.pt carries model + EMA + optimiser
    # (~8 GB), which does not fit beside the built model on a 16 GB card.
    # load_state_dict copies to the parameters' device anyway.
    ckpt = load_checkpoint(load_dir, "cpu")

    # A finetune checkpoint already contains the LoRA adapters, so the wrappers
    # must exist before loading it. A pretrained checkpoint does not, so there
    # the adapters go on afterwards — wrapping renames `...qkv_proj.weight` to
    # `...qkv_proj.base.weight`.
    use_lora = args.lora_rank > 0
    if use_lora and resume_finetune:
        # The finetune checkpoint holds only the trainable tensors; the frozen
        # ones come from the pretrained run, and must land before LoRA wraps the
        # linears and renames them.
        if ckpt.get("trainable_only", False):
            base = load_checkpoint(args.pretrained_dir, "cpu")
            load_state_dict_shape_filtered(model, base["model"], verbose=master_process)
            del base
        apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout,
                   verbose=master_process)

    summary = load_state_dict_shape_filtered(model, ckpt["model"], verbose=master_process)

    if use_lora and not resume_finetune:
        apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout,
                   verbose=master_process)
    if use_lora:
        freeze_for_lora(model, verbose=master_process)

    # The EMA snapshots the trainable parameter set when the model is built, so
    # it must be rebuilt whenever that set changed — freezing for LoRA, or a
    # resized projection whose stored shadow weights no longer fit. When neither
    # happened, the pretrained EMA still matches and is carried over.
    if resume_finetune:
        rebuild_ema(model)
        model.ema.load_state_dict(ckpt["ema"])
    else:
        if use_lora or summary["mismatched"]:
            rebuild_ema(model)
        else:
            model.ema.load_state_dict(ckpt["ema"])
        model.ema.ema.optimization_step = 0

    if cfg.use_compile:
        model = torch.compile(model)

    if cfg.use_activation_checkpointing:
        for module in model.modules():
            if isinstance(module, STAttentionBlock):
                module._use_checkpoint = True
        enable_activation_checkpointing(model, master_process)

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    if use_lora:
        optimizer = configure_optimizers_split(
            raw_model, cfg.training.weight_decay, cfg.training.max_lr,
            tuple(cfg.training.betas), args.projection_lr_mult, verbose=master_process)
    else:
        optimizer = raw_model.configure_optimizers(
            weight_decay=cfg.training.weight_decay,
            learning_rate=cfg.training.max_lr,
            betas=tuple(cfg.training.betas),
            device_type="cuda",
        )
    if resume_finetune:
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resuming finetune from step {step}")
    else:
        step = 0
        print(f"Starting finetune from pretrained checkpoint at {args.pretrained_dir}")

    del ckpt                      # several GB of host memory, no longer needed
    torch.cuda.empty_cache()

    def save_on_preempt(signum, frame):
        if master_process:
            save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step, trainable_only=use_lora)
        if ddp:
            dist.barrier()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, save_on_preempt)

    if master_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in WEAVER: {num_params}")
        if cfg.use_wandb:
            import wandb
            entity = getattr(cfg.wandb, "entity", None)
            if entity in ("", "null", "none", "None"):
                entity = None      # fall back to the logged-in account's default
            wandb.init(project=cfg.wandb.project, entity=entity, sync_tensorboard=False,
                       config=cfg_dict, name=f"{cfg.exp_name}",
                       group=f"{cfg.dataset.name}/{cfg.exp_name}")
            wandb.summary["trainable_params"] = num_params

    train_iter = iter(cycle(train_loader))
    valid_video_iter = iter(cycle(valid_video_loader))
    data = next(train_iter)
    data = move_tensors_to_device(data, device="cuda")

    while step <= cfg.training.max_steps:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        optimizer.zero_grad()
        loss_accum = 0

        for accum_idx in range(cfg.training.gradient_accumulation_steps):
            data = move_tensors_to_device(next(train_iter), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total_loss, loss_log = model(
                    obs=data["obs"],
                    actions=data["actions"],
                    tasks=data["task"],
                    gt_rewards=data["rewards"],
                    memory=data.get("memory", None),
                    objects=data.get("objects", None),
                    object_valid=data.get("object_valid", None),
                )
            total_loss = total_loss / cfg.training.gradient_accumulation_steps
            loss_accum += total_loss.detach()
            if ddp:
                model.require_backward_grad_sync = accum_idx == cfg.training.gradient_accumulation_steps - 1
            total_loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps, cfg.training.max_lr, cfg.training.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        optimizer.step()
        raw_model.ema.update()

        if step % cfg.log_freq == 0 and master_process:
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            # Only present when the auxiliary object head is enabled.
            obj_str = (f" | Obj: {loss_log['aux/objects'].item():.6f}"
                       if 'aux/objects' in loss_log else "")
            print(
                f"Step: {step} | Time: {time.time() - t0:.3f}s"
                f" | Total: {loss_accum.item():.6f}"
                f" | Flow: {loss_log['flow/Total Loss'].item():.6f}"
                f" | RM: {loss_log['RM loss'].item():.6f}"
                f" | V: {loss_log['Critic Loss'].item():.6f}"
                f" | Dec: {loss_log['decoder/Total Loss'].item():.6f}"
                f"{obj_str}"
                f" | LR: {lr:.6f} | Norm: {norm:.6f}"
                f" | Mem(GB) alloc: {mem_after:.2f} peak: {mem_peak:.2f} reserved: {mem_reserved:.2f} before: {mem_before:.2f}"
            )
            if cfg.use_wandb:
                import wandb
                wandb.log({k: v.item() for k, v in loss_log.items()}, step=step)

        if cfg.valid_log_freq > 0 and (step == 0 or step % cfg.valid_log_freq == 0):
            model.eval()
            raw_model.eval()
            if hasattr(valid_loader.sampler, "set_epoch"):
                valid_loader.sampler.set_epoch(step)
            valid_metrics = evaluate_with_metrics_ddp(
                model=model,
                raw_model=raw_model,
                val_dataloader=iter(valid_loader),
                img_keys=img_keys,
                device="cuda",
                master_process=master_process,
                world_size=world_size,
                horizon=cfg.eval_horizon,
                bootstrap=cfg.eval_bootstrap,
            )
            if master_process:
                metrics = {f"valid/{k}": v for k, v in valid_metrics.items()}
                print(f"Step: {step} | Valid Metrics: {metrics}")
                if cfg.use_wandb:
                    import wandb
                    wandb.log(metrics, step=step)
            raw_model.train()
            model.train()

        if cfg.save_model and step % cfg.ckpt_freq == 0 and master_process:
            save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step, trainable_only=use_lora)
            if step % 5000 == 0:
                save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step, suffix=str(step), trainable_only=use_lora)

        if cfg.video_log_freq > 0 and step % cfg.video_log_freq == 0 and master_process:
            save_validation_videos(raw_model, valid_video_iter, cfg, img_keys, vid_dir, step,
                                   use_wandb=cfg.use_wandb)

        if ddp:
            dist.barrier()
        step += 1

    if cfg.save_model and master_process:
        save_checkpoint(chkpt_save_dir, model=getattr(raw_model, "_orig_mod", raw_model) if cfg.use_compile else raw_model, optimizer=optimizer, cfg=cfg, step=step, trainable_only=use_lora)
    if ddp:
        destroy_process_group()
    print("Finetuning completed successfully.")


if __name__ == "__main__":
    main()
