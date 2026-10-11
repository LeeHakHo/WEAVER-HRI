#!/usr/bin/env python3
"""Drive the finetuned world model from the keyboard.

Seeds the model with a few real frames from the validation split, then imagines
everything after that.  Both agents live in the same 18-D space — three
base-frame points each, an origin and the two surfaces that close on an object —
so the robot and the human hand take the *same* five controls:

    move it (x, y, z), open/close it, turn it.

Nothing touches a real robot; this is the world model dreaming.

Controls
--------
                 +X   -X   +Y   -Y   +Z    -Z     open/close   turn
    robot        W    S    A    D    Q     E      1 / 2        Z / X
    hand         Up   Down Left Rght PgUp  PgDn   [ / ]        , / .

    Tab        highlight the other agent in the readout
    C          switch camera
    Backspace  re-seed from the real clip
    V          start/stop recording to mp4
    Esc        quit

Example
-------
    .venv/bin/python -m weaver.interactive \
        --checkpoint runs/weaver_hri/logs/chkpts_hri \
        --overrides dataset.path=data/weaver_hri
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path

import numpy as np
import torch

from .datasets.droid import PrecomputedDroid
from .finetune_utils import load_finetuned
from .generate_views import build_model, load_eval_config
from .wm.streaming import StreamingRollout

# 18-D layout, mirroring tools/hri_common.py: two agents x (origin, contact, contact).
AGENTS = {"robot": slice(0, 9), "hand": slice(9, 18)}

# Held keys command a *sustained* motion, so the size that matters is not the
# per-frame step but where it lands after a few seconds of holding. Over 48
# frames (9.6 s) the training set moves an origin by a median of 11.5 cm, a p90
# of 21.4 cm and a p99 of 29.5 cm. The earlier 0.8 cm/frame reaches 38.4 cm over
# that window -- past the p99 -- and the model tracks it at only 0.63-0.78 of
# what was asked. 0.3 cm/frame lands at 14.4 cm, just above the median, and
# tracking rises to 0.80-1.18. The per-frame figure looked reasonable on its own
# (the p85 of single-frame motion); it was holding it down that went
# out of distribution.
# What the data calls "not moving" is not a zero gap between the commanded and
# the observed pose. On the 33% of training frames where the arm's origin moves
# less than 0.1 mm, the command still sits a steady (-0.05, 0.16, 0.58) cm from
# the observation at the origin, and the commanded jaws are about 1 cm wider
# than the measured ones -- a standing bias of the teleoperation rig, not
# tracking lag. Commanding the observed pose exactly therefore reads as "go
# 0.6 cm lower and close a little", which is why an empty command used to walk
# the arm 6 cm. The human hand has no such bias (0.013-0.018 cm) because its
# command is defined as its own next observed pose.
REST_OFFSET_PATH = "rest_offset.npy"   # resolved under the dataset root

STEP_POS = 0.003     # metres per 5 fps frame -> 14.4 cm over 9.6 s of holding
STEP_OPEN = 0.002    # metres of extra separation per frame
STEP_TURN = 0.023    # radians per frame


# One control table, keyed by the browser's KeyboardEvent.key names. Both
# frontends (pygame window, browser) map their own key codes onto these, so the
# bindings live in exactly one place.
CONTROLS = {
    # robot
    "w": ("robot", "move", np.array([+1.0, 0.0, 0.0])),
    "s": ("robot", "move", np.array([-1.0, 0.0, 0.0])),
    "a": ("robot", "move", np.array([0.0, +1.0, 0.0])),
    "d": ("robot", "move", np.array([0.0, -1.0, 0.0])),
    "q": ("robot", "move", np.array([0.0, 0.0, +1.0])),
    "e": ("robot", "move", np.array([0.0, 0.0, -1.0])),
    "1": ("robot", "open", +1.0),
    "2": ("robot", "open", -1.0),
    "z": ("robot", "turn", +1.0),
    "x": ("robot", "turn", -1.0),
    # hand
    "ArrowUp":    ("hand", "move", np.array([+1.0, 0.0, 0.0])),
    "ArrowDown":  ("hand", "move", np.array([-1.0, 0.0, 0.0])),
    "ArrowLeft":  ("hand", "move", np.array([0.0, +1.0, 0.0])),
    "ArrowRight": ("hand", "move", np.array([0.0, -1.0, 0.0])),
    "PageUp":     ("hand", "move", np.array([0.0, 0.0, +1.0])),
    "PageDown":   ("hand", "move", np.array([0.0, 0.0, -1.0])),
    "[": ("hand", "open", +1.0),
    "]": ("hand", "open", -1.0),
    ",": ("hand", "turn", +1.0),
    ".": ("hand", "turn", -1.0),
}


def accumulate(held) -> dict:
    """Held key names -> per-agent {move, open, turn} commands."""
    cmd = {a: {"move": np.zeros(3), "open": 0.0, "turn": 0.0} for a in AGENTS}
    for name in held:
        spec = CONTROLS.get(name)
        if spec is None:
            continue
        agent, kind, amount = spec
        if kind == "move":
            cmd[agent]["move"] = cmd[agent]["move"] + amount * STEP_POS
        elif kind == "open":
            cmd[agent]["open"] += amount * STEP_OPEN
        else:
            cmd[agent]["turn"] += amount * STEP_TURN
    return cmd


def pygame_keymap(pygame):
    """pygame key constant -> the canonical key name used by CONTROLS."""
    K = pygame
    return {
        K.K_w: "w", K.K_s: "s", K.K_a: "a", K.K_d: "d", K.K_q: "q", K.K_e: "e",
        K.K_1: "1", K.K_2: "2", K.K_z: "z", K.K_x: "x",
        K.K_UP: "ArrowUp", K.K_DOWN: "ArrowDown",
        K.K_LEFT: "ArrowLeft", K.K_RIGHT: "ArrowRight",
        K.K_PAGEUP: "PageUp", K.K_PAGEDOWN: "PageDown",
        K.K_LEFTBRACKET: "[", K.K_RIGHTBRACKET: "]",
        K.K_COMMA: ",", K.K_PERIOD: ".",
    }


def apply_controls(points: np.ndarray, move: np.ndarray, open_amt: float,
                   turn: float) -> np.ndarray:
    """(3, 3) origin/contact_a/contact_b -> where the controls want them next.

    Translation moves the whole triple; opening slides the two contacts apart
    along the line between them, keeping their midpoint; turning rotates the
    contacts about the origin's vertical axis.  The same three operations mean
    the same thing for a parallel-jaw gripper and for a thumb and index finger,
    which is the point of the shared representation.
    """
    out = points.astype(np.float64).copy()
    o, a, b = out

    if open_amt:
        axis = a - b
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        a = a + axis * (open_amt / 2)
        b = b - axis * (open_amt / 2)

    if turn:
        c, s = np.cos(turn), np.sin(turn)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        a = o + rot @ (a - o)
        b = o + rot @ (b - o)

    return np.stack([o, a, b]) + move


def load_rest_offset(root: str | Path, device: str) -> torch.Tensor:
    """The commanded-minus-observed bias the rig holds when nothing is moving."""
    p = Path(root) / REST_OFFSET_PATH
    if not p.exists():
        return torch.zeros(18, dtype=torch.float32, device=device)
    return torch.tensor(np.load(p), dtype=torch.float32, device=device)


class Normaliser:
    """Physical units <-> the normalised space the model is trained in."""

    def __init__(self, path: str | Path, device: str, root: str | Path | None = None):
        # PrecomputedDroid resolves a relative norm_stats_path under the dataset
        # root, so the same config value has to mean the same file here or the
        # two halves of a rollout end up normalised differently.
        p = Path(path)
        if not p.is_absolute() and root is not None and not p.exists():
            p = Path(root) / p
        stats = json.loads(p.read_text())["norm_stats"]
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.s_mean, self.s_std = t(stats["state"]["mean"]), t(stats["state"]["std"])
        self.a_mean, self.a_std = t(stats["actions"]["mean"]), t(stats["actions"]["std"])

    def norm_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.a_mean) / self.a_std

    def denorm_state(self, s: torch.Tensor) -> torch.Tensor:
        return s * self.s_std + self.s_mean


def load_seed_clip(cfg, img_keys, traj_index: int, n_seed: int, device: str):
    """Real latents / states / actions for the first ``n_seed`` frames of a val clip."""
    encoder_type = "sd3" if "stable-diffusion-3" in cfg.im_encoder.name else "svd"
    dataset = PrecomputedDroid(
        root=cfg.dataset.path, split="val", horizon=cfg.horizon, img_keys=img_keys,
        relabel_actions=cfg.dataset.relabel_actions, normalize=cfg.dataset.normalize,
        norm_stats_path=getattr(cfg.dataset, "norm_stats_path", None),
        cache_trajectories=False, return_language=True, return_video_frames=False,
        encoder_type=encoder_type, n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory, n_history=cfg.n_history, eval_mode=True,
        annotation_dir=cfg.dataset.annotation_dir,
    )
    ti = dataset.valid_trajectories[traj_index % len(dataset.valid_trajectories)]
    traj = dataset.trajectories[ti]
    traj.load()
    n_seed = min(n_seed, len(traj), len(dataset._states[ti]))

    obs = {
        f"{k}_features": torch.from_numpy(
            np.ascontiguousarray(traj.latents[k][:n_seed])
        ).float().unsqueeze(0).to(device)
        for k in img_keys
    }
    nd = dataset.norm_dict if dataset.normalize else None
    states = dataset._states[ti][:n_seed].clone()
    actions = dataset._actions[ti][:n_seed].clone()
    if nd:
        states = (states - nd["states"]["mean"]) / nd["states"]["std"]
        actions = (actions - nd["actions"]["mean"]) / nd["actions"]["std"]
    obs["states"] = states.unsqueeze(0).to(device)
    return obs, actions.unsqueeze(0).to(device), traj.traj_id


def to_surface(pygame, frame: torch.Tensor, scale: int):
    """(C, H, W) in [0,1] -> a scaled pygame surface."""
    img = (frame.clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
    img = np.transpose(img, (2, 1, 0))      # pygame wants (W, H, C)
    surf = pygame.surfarray.make_surface(img)
    if scale != 1:
        surf = pygame.transform.scale(surf, (img.shape[0] * scale, img.shape[1] * scale))
    return surf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="directory holding checkpoint.pt")
    ap.add_argument("--checkpoint-name", default="checkpoint.pt")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--n-seed", type=int, default=None,
                    help="real frames used to seed history+memory (default: full memory span)")
    ap.add_argument("--val-steps", type=int, default=8, help="flow denoising steps")
    ap.add_argument("--scale", type=int, default=2, help="display magnification")
    ap.add_argument("--camera", default="exterior_1_left",
                    help="camera shown first; the rest stay one keypress away")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record-dir", type=Path, default=Path("data/interactive"))
    args = ap.parse_args()

    import pygame

    overrides = list(args.overrides) + [f"model.val_steps={args.val_steps}",
                                        "inference.pyramid_stagger_width=0"]
    cfg = load_eval_config(args.checkpoint, overrides)
    model, img_keys = build_model(cfg, args.device)
    load_finetuned(model, args.checkpoint, cfg,
                   checkpoint_name=args.checkpoint_name, device=args.device)
    model.eval()
    # The rollout buffer holds latents, including for the memory frames, so the
    # model must read features rather than re-encoding images.
    model._use_precomputed_features = True

    norm = Normaliser(cfg.dataset.norm_stats_path, args.device, cfg.dataset.path)
    n_seed = args.n_seed or max(cfg.n_history, cfg.n_memory_frames * cfg.t_memory + 1)
    obs, seed_actions, traj_id = load_seed_clip(cfg, img_keys, args.traj_index,
                                                n_seed, args.device)
    print(f"seeded from val trajectory {traj_id} with {obs['states'].shape[1]} real frames")

    roll = StreamingRollout(model, horizon=cfg.horizon)
    roll.seed(obs, seed_actions)
    bootstrap = roll.bootstrap

    pygame.init()
    keymap = pygame_keymap(pygame)
    cam = img_keys.index(args.camera) if args.camera in img_keys else 0
    panel_w, panel_h = cfg.dataset.image_size[1], cfg.dataset.image_size[0]
    screen = pygame.display.set_mode((panel_w * args.scale, panel_h * args.scale + 92))
    pygame.display.set_caption("WEAVER — keyboard-driven dream")
    font = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()

    focus = "robot"
    recording: list[np.ndarray] = []
    is_recording = False
    target = norm.denorm_state(roll.latest_states(1))[0, 0].cpu().numpy()

    def collect_actions() -> torch.Tensor:
        """Build (1, bootstrap, 18) normalised actions from the keys held down.

        The action is the commanded *pose* of the three points, not a
        displacement, and the demo owns that pose: the key moves `target`,
        which starts at the seed. Re-reading the model's own predicted state
        each chunk instead would make "no key" mean "go to wherever you just
        drifted" -- nothing pulls it back, and the arm wanders under an empty
        command. It is ramped across the chunk so a held key gives continuous
        motion rather than one step and a hold.
        """
        nonlocal target
        pressed = pygame.key.get_pressed()
        held = [name for code, name in keymap.items() if pressed[code]]
        cmd = accumulate(held)

        step = np.zeros(18)
        for agent, sl in AGENTS.items():
            pts = target[sl].reshape(3, 3)
            tgt = apply_controls(pts, cmd[agent]["move"], cmd[agent]["open"],
                                 cmd[agent]["turn"])
            step[sl] = (tgt - pts).reshape(9)
        ramp = np.arange(1, bootstrap + 1)[:, None] * step[None, :]
        chunk_cmd = target[None, :] + ramp
        target = chunk_cmd[-1].copy()
        a = torch.tensor(chunk_cmd, dtype=torch.float32, device=args.device)
        return norm.norm_action(a).view(1, bootstrap, 18)

    # Generation runs off the main thread so the window keeps taking input while
    # the next chunk is denoising; CUDA releases the GIL for the duration.
    worker_result: queue.Queue = queue.Queue()

    def worker(actions):
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                worker_result.put(("ok", roll.step(actions)))
        except Exception as exc:                      # re-raised in the main loop
            worker_result.put(("err", exc))

    def request_next():
        threading.Thread(target=worker, args=(collect_actions(),), daemon=True).start()

    request_next()
    frames: dict | None = None
    frame_i = 0
    running = True
    status = "generating..."

    def fmt(agent: str, st: np.ndarray) -> str:
        pts = st[AGENTS[agent]].reshape(3, 3)
        opening = np.linalg.norm(pts[1] - pts[2]) * 100
        name = agent.upper() if focus == agent else agent
        return (f"[{name:5s}] origin {pts[0][0]:+.3f} {pts[0][1]:+.3f} {pts[0][2]:+.3f}"
                f"   opening {opening:5.1f} cm")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    focus = "hand" if focus == "robot" else "robot"
                elif event.key == pygame.K_c:
                    cam = (cam + 1) % len(img_keys)
                elif event.key == pygame.K_BACKSPACE:
                    roll.seed(obs, seed_actions)
                    target = norm.denorm_state(roll.latest_states(1))[0, 0].cpu().numpy()
                    frames, frame_i = None, 0
                    status = "re-seeded"
                    request_next()
                elif event.key == pygame.K_v:
                    is_recording = not is_recording
                    if not is_recording and recording:
                        args.record_dir.mkdir(parents=True, exist_ok=True)
                        n_prev = len(list(args.record_dir.glob("*.mp4")))
                        out = args.record_dir / f"dream_{traj_id}_{n_prev}.mp4"
                        import imageio
                        imageio.mimwrite(out, np.stack(recording), fps=args.fps)
                        print(f"wrote {out} ({len(recording)} frames)")
                        recording = []

        # Pull a finished chunk and immediately queue the next one.
        if frames is None or frame_i >= bootstrap:
            try:
                kind, payload = worker_result.get_nowait()
            except queue.Empty:
                kind = None
            if kind == "err":
                raise payload
            if kind == "ok":
                frames = payload
                frame_i = 0
                status = f"frame {roll.head}"
                request_next()

        if frames is not None and frame_i < bootstrap:
            frame = frames[img_keys[cam]][0, frame_i]
            screen.fill((18, 18, 20))
            screen.blit(to_surface(pygame, frame, args.scale), (0, 0))

            st = norm.denorm_state(roll.latest_states(1))[0, 0].cpu().numpy()
            y = panel_h * args.scale + 4
            lines = [
                fmt("robot", st),
                fmt("hand", st),
                f"cam {img_keys[cam]}   {status}{'   [REC]' if is_recording else ''}   "
                f"WASD/QE+1,2,Z,X robot   arrows/PgUp,PgDn+[,],',','.' hand   Tab C V Esc",
            ]
            for line in lines:
                screen.blit(font.render(line, True, (220, 220, 220)), (6, y))
                y += 18
            pygame.display.flip()

            if is_recording:
                img = (frame.clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
                recording.append(np.transpose(img, (1, 2, 0)))
            frame_i += 1

        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
