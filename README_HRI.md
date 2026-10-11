# WEAVER on HRI insertion: one world model, two agents

Finetunes WEAVER on `LeeHakHo/play_hri_insertion_{1..6}_eef` (eval episode held
out) so that the **human hand in the video is conditioned on in the same action
space as the robot**, and ships a keyboard demo that drives both.

The datasets are LeRobot v2.1 recordings of a Trossen stationary arm doing
"insert cube" while a person works alongside it — three RGB cameras plus depth,
30 fps, one long unsegmented episode each.  Stock WEAVER conditions on a single
robot's joints; here the world model is told what *both* agents do, so at
inference you can move either one and watch the consequences.

---

## The 18-D action space: both agents as three points

A gripper and a hand have nothing in common at the joint level, and matching
them as "pose + aperture" still means reconciling a palm frame with a
parallel-jaw frame — two different conventions pretending to be one.  So each
agent is reduced to the same three points instead: an origin, and the two
surfaces that close on an object.

```
              origin            contact_a        contact_b
   robot      ee_position       jaw tip +        jaw tip -
   hand       wrist             thumb tip        index tip

idx  0.. 8    robot  origin(3)  contact_a(3)  contact_b(3)
idx  9..17    hand   origin(3)  contact_a(3)  contact_b(3)
```

Everything is a position in the **robot base frame**, in metres —
`meta/camera_calibration.json` states that frame is the one the episodes' own
FK/EEF features live in, which is what lets the two agents be compared at all.
`states[t]` holds the points; `actions[t] = states[t+1] - states[t]` at the 5 fps
video rate.

What this buys:

* **the same number means the same thing for both.**  `‖contact_a − contact_b‖`
  is the opening, in centimetres, for the gripper and for the hand alike; the
  origin is where the agent is; the triangle is how it is posed.
* **no angles**, so no Euler convention, no wrapping, and a plain subtraction
  for the action.
* **no invented constants.**  The previous design normalised the hand's pinch
  onto [0, 1] with a hand-picked 3-14 cm range; there is nothing to pick here.
* **every channel is a point you can see in the image**, which is what a video
  world model can actually be held to.

The one thing three points cannot express is a rotation about the line through
the two contacts — and in this data there is nothing there to lose: the robot's
end-effector axes stay within 1.6° (p95) of their own mean for a whole episode,
with a roll spread of 0.1-0.8°.  It keeps a fixed wrist pose and moves in xyz.

The robot's jaw tips are not recorded, only `ee_position`, `ee_euler_xyz` and a
single `obs_right_gripper` value.  Overlaying candidates on cam_high pins the
geometry down (`jaw_tips` in `tools/hri_common.py`): the jaws separate along the
EEF y axis, the tips sit at the origin's x, and `obs_right_gripper` is the
*half* separation — at 0.0363 the two projected points land exactly on the two
fingertips, and at 0 they collapse onto the closed gripper.

---

## Pipeline

### Phase 1 — hand tracks (`tools/run_phase1.sh`)

`cam_high` (top-down) is the view where the hand is big and unoccluded, so the
whole hand track comes from it.

1. **`tools/run_wilor.py`** (runs in `.venv-wilor`) — WiLoR gives 21 metric MANO
   joints, their 2-D projections and a hand bounding box per frame, every 6th
   frame (30 fps → the 5 fps WEAVER was pretrained at).  The detector runs at a
   deliberately low threshold to catch frames where the hand is half-hidden
   behind the object it holds; `pick_hand()` then screens out what that lets in —
   the robot gripper (rejected by position: it lives on the right of the frame)
   and junk boxes (rejected by area).
2. **`tools/extract_hand_tracks.py`** (runs in `.venv`) — lifts the wrist, thumb
   tip and index tip to the base frame. All three take the identical path, 2-D
   keypoint plus depth, so nothing depends on MANO's 3-D pose fit:
   - *Scale* from MANO seeds the lookup: the wrist→middle-MCP segment has a
     known metric length and a measured pixel length. WiLoR's own `pred_cam_t`
     is unusable here — it assumes focal 5000 against this camera's real 604,
     so it reports the hand ~18 m away.
   - *Depth* then measures each point. The colour and depth streams have
     different intrinsics **and** different extrinsics, so each colour ray is
     walked into the depth camera, read, and re-intersected with the ray rather
     than indexing the depth image at the colour pixel. This corrects the
     monocular estimate by ~20 cm — it is not a refinement, it is the
     measurement.
   - *Cross-check*: MANO knows the metric thumb-index distance independently of
     the depth camera. The two disagree by a median of **1.1-1.5 cm** across
     episodes, which is what makes the depth-derived points believable.
   - *Handedness*: WiLoR's left/right call flickers on 10-20% of frames and a
     wrong call mirrors the joints, which can change which finger a joint index
     lands on. Those frames survive only when all three points stay within 5 cm
     of the confidently-tracked path — the test that separates a mirrored call
     of the same hand from the detector jumping to the person's other hand.
   - Frames with a point outside the workspace box are dropped, gaps
     interpolated and Savitzky-Golay smoothed.

   Result per episode: **70-89% of frames with all three points measured**,
   gaps median 1 frame.
3. **`tools/viz_hand_tracks.py`** — the check that matters. It draws *both*
   agents' three points, and the robot's come straight from the dataset without
   passing through this pipeline. When its two jaw points land on the two
   fingertips and collapse as the gripper closes, the calibration, the
   base-frame convention, the projection code and the jaw geometry are all
   confirmed at once — and the hand, which takes the identical path, is
   trustworthy too.

### Phase 2 — WEAVER dataset (`tools/run_phase2.sh`)

`tools/lerobot_to_weaver.py` writes the layout `datasets/README.md` specifies,
so the stock dataloader reads it unchanged:

```
data/weaver_hri/
  annotation_rewards/{train,val}/<id>.json    states, actions, text features
  videos/{train,val}/<id>.mp4                 192x960: cam_high|cam_low|cam_wrist
  latents/{train,val}/<id>_sd3.npz            (3, T, 60, 256) float16
  norm_stats_relabel.json
```

Cameras map onto WEAVER's fixed slots as cam_high→`exterior_1_left`,
cam_low→`exterior_2_left`, cam_right_wrist→`wrist_left`; training uses the same
two views the released checkpoint was trained with.  Each ~3000-frame episode is
cut into 200-frame segments — one trajectory per episode would give the sampler
nothing to shuffle.

Episodes 1-6 → `train`, the eval episode → `val`.

Reward is a constant: this is play data with no success labels, and the reward
and critic heads are switched off for the finetune, so the RoboMeter labelling
step of the original pipeline is skipped entirely.

### Phase 3 — finetune (`scripts/finetune_local.sh`)

The released checkpoint is 8-D (Franka joints + gripper) and ~915 M parameters
in the world model alone; the target is 18-D on a 16 GB card.  `weaver/finetune_utils.py`
handles both:

- `load_state_dict_shape_filtered` loads the 661 tensors that fit and reports
  the 9 that do not — the first layer of each state/action MLP and the state
  read-out — leaving them at fresh init.
- `apply_lora` / `freeze_for_lora` wrap the 128 linears in the transformer
  blocks (rank 32) and train the projections densely: **33.7 M of 1.16 B
  trainable (2.9%)**. Model, adapters, EMA and optimiser together occupy 4.8 GB
  and a training step peaks at 7.5 GB of the card's 15.6 GB.
- The checkpoint is loaded onto the CPU: `checkpoint.pt` carries model + EMA +
  optimiser (~8 GB) and will not fit beside the built model on a 16 GB card.
- The EMA is rebuilt, since it snapshots the trainable set at construction and
  that set has changed.

`n_states`/`n_actions` are deliberately excluded from the config keys inherited
from the pretrained run — changing them is the point.

### Phase 4 — evaluation (`scripts/eval_local.sh`)

FID / FVD / LPIPS on the held-out episode, plus the test those metrics cannot
do: `tools/counterfactual_rollout.py` replays the same clip from the same seed
four times, varying only which half of the action vector is live
(`full` / `robot` / `hand` / `frozen`), and tiles the results.  "Hold still" is
a zero delta in metres, so with this representation it is literally zero motion.
A model that ignored the hand channels would still score well on FVD; it would
visibly fail here.

### Phase 5 — keyboard demo (`weaver/interactive.py`)

Seeds from a few real validation frames, then imagines everything.
`weaver/wm/streaming.py` turns WEAVER's chunked memory rollout — which normally
wants the whole action sequence up front — into `seed()` / `step()`, so actions
can arrive from the keyboard. Generation runs on a worker thread so the window
stays responsive while a chunk denoises.

Because both agents are the same three points, they take the same five
controls — move it, open it, turn it:

```
             +X   -X   +Y   -Y   +Z    -Z     open/close   turn
 robot       W    S    A    D    Q     E      1 / 2        Z / X
 hand        Up   Down Left Rght PgUp  PgDn   [ / ]        , / .
 Tab focus   C camera   Backspace re-seed   V record   Esc quit
```

The target points are computed from the model's *own* latest predicted state,
so what "open" and "turn" mean stays consistent with whatever the world model
currently believes rather than drifting from an integrated command.

---

## Running it

```bash
uv venv --python 3.11 && uv sync                      # .venv
uv venv --python 3.10 .venv-wilor                     # WiLoR needs its own
uv pip install --python .venv-wilor/bin/python pip setuptools wheel
uv pip install --python .venv-wilor/bin/python --no-build-isolation "numpy<1.24" chumpy
uv pip install --python .venv-wilor/bin/python --no-build-isolation \
    "git+https://github.com/warmshao/WiLoR-mini" dill

.venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('arnavkj1995/WEAVER', allow_patterns=['WEAVER/*'], local_dir='checkpoints')"

bash tools/run_phase1.sh      # ~10 min/episode; review data/viz/*_overlay.mp4
bash tools/run_phase2.sh      # ~5 min/episode
bash scripts/finetune_local.sh
bash scripts/eval_local.sh
.venv/bin/python -m weaver.interactive \
    --checkpoint runs/weaver_hri/logs/chkpts_hri \
    --overrides dataset.path=data/weaver_hri
```

MANO is **not** a manual step: WiLoR-mini pulls `MANO_RIGHT.pkl` from its own
public HuggingFace repo. Note that mirror redistributes a model whose licence is
non-commercial research use.

Needs ~35 GB of free disk and a CUDA GPU with 16 GB.

---

## Changes to the upstream repo

| file | change |
|---|---|
| `weaver/datasets/droid.py` | use `states`/`actions` from the annotation when present; reward handling split into `_preprocess_rewards` |
| `weaver/finetune.py` | shape-filtered checkpoint load, LoRA flags, EMA rebuild, `n_states`/`n_actions` no longer inherited |
| `weaver/generate_views.py` | rebuild LoRA wrappers before loading a LoRA checkpoint |
| `weaver/finetune_utils.py` | *new* — filtered loading, LoRA, EMA rebuild |
| `weaver/wm/streaming.py` | *new* — `StreamingRollout` |
| `weaver/interactive.py` | *new* — keyboard demo |
| `tools/`, `scripts/*_local.sh` | *new* — the data pipeline and single-GPU launchers |

Everything else, including `generate_videos_full_memory`, is untouched: it is
what evaluation runs on, and the streaming path is additive rather than a
refactor of it.

---

## Known limits

- **One hour of data, one task, one scene.** Long rollouts will drift; the
  demo is meant for the few-second horizon.
- **Domain gap.** The pretrained model saw Franka arms in DROID scenes; the
  Trossen arm's appearance has to be learned almost from scratch by 34 M
  adapter parameters.
- **The hand track is a reconstruction, not ground truth.** 70-89% of frames per
  episode have all three points measured; the rest is interpolation.
  `hand_valid` is kept in every annotation for anyone who wants to mask the
  loss on it.
- **Only one hand is tracked.** In stretches where the person uses both hands,
  `pick_hand` takes the largest box in the region of interest, which can
  alternate between them. The position-continuity test removes those frames
  rather than following the wrong hand, so they become gaps. Carrying *every*
  detection through to `extract_hand_tracks.py` and tracking by continuity from
  the start would do better; it would cost a full re-run of Phase 1.
- **A fingertip's depth is harder to read than a wrist's.** The patch median can
  catch the grasped object instead; a reading more than 20 cm from the wrist's
  falls back to the wrist's depth, which costs at most the hand's own depth
  extent under this top-down camera.
