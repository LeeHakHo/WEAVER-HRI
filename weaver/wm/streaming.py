"""Chunk-at-a-time rollout, for driving the world model interactively.

``WEAVER.generate_videos_full_memory`` needs the entire action sequence up
front: it is written for replaying a recorded trajectory.  Keyboard control
needs the opposite order — generate a few frames, show them, then read the next
actions from whatever the user pressed.

``StreamingRollout`` keeps the same frame buffer and the same
``t_memory``-spaced memory lookups as that function, but exposes them as
``seed()`` / ``step()``.  It deliberately does not modify the batch path: that
one is used by evaluation and the two have different enough control flow that
sharing the loop would complicate both.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class StreamingRollout:
    """Autoregressive latent rollout fed one action chunk at a time.

    Usage::

        roll = StreamingRollout(model)
        roll.seed(obs, actions)                  # real history from a dataset clip
        while True:
            frames = roll.step(user_actions)     # (B, bootstrap, C, H, W) per camera

    ``obs`` holds the real observations for frames ``0..T0-1`` — precomputed
    latents under ``f"{key}_features"`` plus ``states`` — exactly the dict the
    dataloader produces.  ``actions[:, t]`` is the action taken *at* frame ``t``,
    i.e. the one that produces frame ``t+1``; that is the dataset's convention.
    """

    def __init__(
        self,
        model,
        horizon: Optional[int] = None,
        bootstrap: Optional[int] = None,
        capacity: int = 2048,
        use_cache: bool = True,
    ):
        self.model = model
        self.horizon = horizon or model._n_horizon
        self.bootstrap = bootstrap or self.horizon
        if not 1 <= self.bootstrap <= self.horizon:
            raise ValueError("bootstrap must be in [1, horizon]")
        self.capacity = capacity
        self.use_cache = use_cache

        self.n_hist = model._n_history
        self.n_memory = model._n_memory_frames if model._use_memory else 0
        self.t_memory = model._t_memory
        self.buf_offset = self.n_memory * self.t_memory

        self.frame_buffer: Dict[str, torch.Tensor] = {}
        self.action_buffer: Optional[torch.Tensor] = None
        self.head = 0          # absolute index of the next frame to predict

    # -- setup ---------------------------------------------------------------

    @torch.no_grad()
    def seed(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> None:
        """Fill the buffers with real observations and the actions that led to them."""
        x1 = self.model.encode_obs(obs)
        ref = next(iter(x1.values()))
        B, T0 = ref.shape[:2]
        if T0 < 1:
            raise ValueError("seed needs at least one real frame")
        device, dtype = ref.device, ref.dtype

        total = self.buf_offset + self.capacity
        self.frame_buffer = {
            k: torch.zeros(B, total, *v.shape[2:], device=device, dtype=v.dtype)
            for k, v in x1.items()
        }
        for k, v in x1.items():
            self.frame_buffer[k][:, self.buf_offset:self.buf_offset + T0] = v

        A = actions.shape[-1]
        self.action_buffer = torch.zeros(B, total, A, device=device, dtype=actions.dtype)
        n_act = min(actions.shape[1], T0)
        self.action_buffer[:, self.buf_offset:self.buf_offset + n_act] = actions[:, :n_act]

        self.head = T0
        self._B, self._device, self._dtype = B, device, dtype

    # -- internals -----------------------------------------------------------

    def _buf(self, abs_positions) -> list[int]:
        """Clamp absolute frame positions into buffer indices (frame 0 for negatives)."""
        hi = self.capacity - 1
        return [self.buf_offset + max(0, min(int(p), hi)) for p in abs_positions]

    def _memory_tokens(self) -> Optional[torch.Tensor]:
        if self.n_memory == 0:
            return None
        mem_abs = [self.head - (self.n_memory - j) * self.t_memory
                   for j in range(self.n_memory)]
        idx = self._buf(mem_abs)
        mem_obs = {f"{k}_features": self.frame_buffer[k][:, idx]
                   for k in self.model._img_keys}
        mem_obs["states"] = self.frame_buffer["states"][:, idx]
        return self.model.encode_memory_obs(mem_obs)

    # -- stepping ------------------------------------------------------------

    @torch.no_grad()
    def step(self, step_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate the next ``bootstrap`` frames.

        ``step_actions`` is (B, bootstrap, A); ``step_actions[:, j]`` is the
        action that produces frame ``head + j``.  It is stored at buffer index
        ``head - 1 + j``, matching the dataset's "action at frame t produces
        frame t+1" alignment.

        Returns the decoded cameras plus the predicted ``states``, all
        (B, bootstrap, ...).
        """
        if self.action_buffer is None:
            raise RuntimeError("call seed() before step()")
        n_new = step_actions.shape[1]
        if n_new != self.bootstrap:
            raise ValueError(f"expected {self.bootstrap} actions, got {n_new}")
        if self.head + self.bootstrap >= self.capacity:
            raise RuntimeError("rollout exceeded the buffer capacity; re-seed or enlarge it")

        # Store the incoming actions at head-1 .. head+bootstrap-2.
        for j in range(n_new):
            self.action_buffer[:, self._buf([self.head - 1 + j])[0]] = step_actions[:, j]
        # The chunk the model sees runs to head+horizon-1, one slot past the
        # last action above, and the model reads a frame's own slot when it
        # generates that frame. Left at zero, that slot is not "no command":
        # with normalised commanded-pose actions it orders the dataset-mean
        # pose, and the last frame of every chunk jumped ~5 cm toward it.
        # Repeat the last action instead -- "hold" -- until the next step()
        # overwrites it. The batch rollout has the real next action there.
        for p in range(self.head - 1 + n_new, self.head + self.horizon):
            self.action_buffer[:, self._buf([p])[0]] = step_actions[:, -1]

        chunk_start = self.head - self.n_hist
        pos = list(range(chunk_start, self.head + self.horizon))
        idx = self._buf(pos)

        x1_chunk = {}
        for k, buf in self.frame_buffer.items():
            x1_chunk[k] = torch.zeros(
                self._B, self.n_hist + self.horizon, *buf.shape[2:],
                device=buf.device, dtype=buf.dtype,
            )
            x1_chunk[k][:, :self.n_hist] = buf[:, idx[:self.n_hist]]
        actions_chunk = self.action_buffer[:, idx]

        memory_tokens = self._memory_tokens()
        gen = (self.model.generate_latent_rollouts_cached if self.use_cache
               else self.model.generate_latent_rollouts)
        xt = gen(x1_chunk, actions_chunk, memory_tokens=memory_tokens)

        new = {k: v[:, self.n_hist:self.n_hist + self.bootstrap] for k, v in xt.items()}
        store = self._buf(range(self.head, self.head + self.bootstrap))
        for k in self.frame_buffer:
            self.frame_buffer[k][:, store] = new[k]
        self.head += self.bootstrap

        return self.model.decode_obs(new, chunk_size=16)

    @torch.no_grad()
    def latest_states(self, n: int = 1) -> torch.Tensor:
        """The last ``n`` frames' states, as predicted (or seeded)."""
        idx = self._buf(range(self.head - n, self.head))
        return self.frame_buffer["states"][:, idx]
