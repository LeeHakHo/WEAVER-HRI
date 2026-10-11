"""Index bookkeeping of StreamingRollout, checked against a stub model.

The alignment this verifies is easy to get wrong and expensive to debug through
a 915M-parameter model: which actions reach which chunk, which frames the
history window holds, where predictions land in the buffer, and which frames the
sparse memory looks at.  The stub records every call and returns recognisable
latents, so each of those can be asserted exactly.

    .venv/bin/python -m pytest tests/test_streaming.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weaver.wm.streaming import StreamingRollout  # noqa: E402

N_ACTIONS = 14
N_TOKENS, N_DIM = 4, 8
IMG_KEY = "cam"


class StubModel:
    """Stands in for WEAVER, recording what the rollout asks of it.

    Latents are filled with the value of the action that produced them, so the
    test can read a frame back out of the buffer and tell which action made it.
    """

    def __init__(self, n_history=2, n_horizon=8, n_memory_frames=3, t_memory=5):
        self._n_history = n_history
        self._n_horizon = n_horizon
        self._n_memory_frames = n_memory_frames
        self._t_memory = t_memory
        self._use_memory = n_memory_frames > 0
        self._img_keys = [IMG_KEY]
        self.calls = []
        self.memory_calls = []

    def encode_obs(self, obs):
        return {IMG_KEY: obs[f"{IMG_KEY}_features"], "states": obs["states"]}

    def encode_memory_obs(self, memory_obs):
        self.memory_calls.append(memory_obs[f"{IMG_KEY}_features"][..., 0, 0].clone())
        return torch.zeros(memory_obs["states"].shape[0], 1, 1)

    def generate_latent_rollouts_cached(self, x1, actions, memory_tokens=None):
        self.calls.append({"actions": actions.clone(),
                           "history": x1[IMG_KEY][:, :self._n_history, 0, 0].clone()})
        B, T = actions.shape[:2]
        n_hist = self._n_history
        out_img = x1[IMG_KEY].clone()
        out_states = x1["states"].clone()
        # Tag each generated frame with the action that produced it: the action
        # at chunk position n_hist-1+j drives frame n_hist+j.
        for j in range(T - n_hist):
            tag = actions[:, n_hist - 1 + j, 0]
            out_img[:, n_hist + j] = tag[:, None, None]
            out_states[:, n_hist + j] = tag[:, None]
        return {IMG_KEY: out_img, "states": out_states}

    def decode_obs(self, x, chunk_size=0):
        return {IMG_KEY: x[IMG_KEY], "states": x["states"]}


def make_seed(n_seed: int, batch: int = 1):
    """Real frames numbered -1, -2, ... so they never collide with action tags."""
    img = torch.full((batch, n_seed, N_TOKENS, N_DIM), 0.0)
    for t in range(n_seed):
        img[:, t] = -(t + 1)
    obs = {f"{IMG_KEY}_features": img,
           "states": torch.full((batch, n_seed, N_ACTIONS), 0.0)}
    actions = torch.zeros(batch, n_seed, N_ACTIONS)
    return obs, actions


def test_actions_reach_the_frames_they_drive():
    model = StubModel()
    roll = StreamingRollout(model, capacity=128)
    roll.seed(*make_seed(20))
    assert roll.head == 20

    step_actions = torch.arange(1, roll.bootstrap + 1, dtype=torch.float32)
    step_actions = step_actions.view(1, -1, 1).repeat(1, 1, N_ACTIONS)
    out = roll.step(step_actions)

    # Frame head+j must carry the tag of step_actions[:, j].
    assert out[IMG_KEY].shape[1] == roll.bootstrap
    produced = out[IMG_KEY][0, :, 0, 0]
    assert torch.equal(produced, step_actions[0, :, 0])
    assert roll.head == 20 + roll.bootstrap


def test_history_window_follows_the_head():
    model = StubModel()
    roll = StreamingRollout(model, capacity=128)
    roll.seed(*make_seed(20))

    a1 = torch.full((1, roll.bootstrap, N_ACTIONS), 7.0)
    roll.step(a1)
    a2 = torch.full((1, roll.bootstrap, N_ACTIONS), 9.0)
    roll.step(a2)

    # First chunk's history is the last two *real* frames (tagged -20, -19 ->
    # seeded as -(t+1) for t = 18, 19).
    assert torch.equal(model.calls[0]["history"][0], torch.tensor([-19.0, -20.0]))
    # Second chunk's history is the last two frames the first chunk predicted.
    assert torch.equal(model.calls[1]["history"][0], torch.tensor([7.0, 7.0]))


def test_memory_frames_are_t_memory_spaced_and_clamped():
    model = StubModel(n_memory_frames=3, t_memory=5)
    roll = StreamingRollout(model, capacity=128)
    roll.seed(*make_seed(20))
    roll.step(torch.zeros(1, roll.bootstrap, N_ACTIONS))

    # head = 20 -> memory at 5, 10, 15, i.e. seeded frames tagged -6, -11, -16.
    assert torch.equal(model.memory_calls[0][0], torch.tensor([-6.0, -11.0, -16.0]))

    # Near the start every lookup clamps to frame 0.
    model2 = StubModel(n_memory_frames=3, t_memory=5)
    roll2 = StreamingRollout(model2, capacity=128)
    roll2.seed(*make_seed(2))
    roll2.step(torch.zeros(1, roll2.bootstrap, N_ACTIONS))
    assert torch.equal(model2.memory_calls[0][0], torch.tensor([-1.0, -1.0, -1.0]))


def test_action_chunk_covers_history_and_future():
    model = StubModel()
    roll = StreamingRollout(model, capacity=128)
    obs, actions = make_seed(20)
    actions[:, :] = -1.0                       # recognisable "past" actions
    roll.seed(obs, actions)

    step_actions = torch.arange(1, roll.bootstrap + 1, dtype=torch.float32)
    step_actions = step_actions.view(1, -1, 1).repeat(1, 1, N_ACTIONS)
    roll.step(step_actions)

    chunk = model.calls[0]["actions"][0, :, 0]
    n_hist = model._n_history
    assert len(chunk) == n_hist + model._n_horizon
    # Positions [0 .. n_hist-2] are past actions ...
    assert chunk[0] == -1.0
    # ... then the new ones, which drive the frames that get kept ...
    assert torch.equal(chunk[n_hist - 1:n_hist - 1 + roll.bootstrap],
                       step_actions[0, :, 0])
    # ... and the tail repeats the last action. It must not stay zero: the
    # model reads a frame's own slot, and a zero there is a command to the
    # dataset-mean pose once actions are normalised absolute poses.
    assert torch.all(chunk[n_hist - 1 + roll.bootstrap:] == step_actions[0, -1, 0])


def test_latest_states_tracks_the_head():
    model = StubModel()
    roll = StreamingRollout(model, capacity=128)
    roll.seed(*make_seed(20))
    roll.step(torch.full((1, roll.bootstrap, N_ACTIONS), 3.0))
    assert roll.latest_states(1)[0, 0, 0].item() == 3.0


def test_rejects_wrong_action_count_and_overflow():
    model = StubModel()
    roll = StreamingRollout(model, capacity=128)
    roll.seed(*make_seed(20))
    with pytest.raises(ValueError):
        roll.step(torch.zeros(1, roll.bootstrap + 1, N_ACTIONS))

    small = StreamingRollout(StubModel(), capacity=24)
    small.seed(*make_seed(20))
    with pytest.raises(RuntimeError):
        small.step(torch.zeros(1, small.bootstrap, N_ACTIONS))


def test_step_before_seed_is_an_error():
    roll = StreamingRollout(StubModel(), capacity=32)
    with pytest.raises(RuntimeError):
        roll.step(torch.zeros(1, roll.bootstrap, N_ACTIONS))


# --- the shared three-point controls ---------------------------------------
# Both agents are {origin, contact_a, contact_b}, so one set of operations has to
# mean the same thing for a parallel-jaw gripper and for a thumb and index
# finger. These pin down that meaning.

def _triple():
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, -0.05, 0.0]])


def test_opening_changes_separation_and_keeps_the_midpoint():
    from weaver.interactive import apply_controls
    pts = _triple()
    out = apply_controls(pts, np.zeros(3), 0.02, 0.0)
    assert np.isclose(np.linalg.norm(out[1] - out[2]), 0.12)
    assert np.allclose((out[1] + out[2]) / 2, (pts[1] + pts[2]) / 2)
    assert np.allclose(out[0], pts[0])


def test_moving_translates_all_three_and_leaves_the_opening():
    from weaver.interactive import apply_controls
    pts = _triple()
    out = apply_controls(pts, np.array([0.1, -0.03, 0.02]), 0.0, 0.0)
    assert np.allclose(out - pts, np.array([0.1, -0.03, 0.02]))
    assert np.isclose(np.linalg.norm(out[1] - out[2]), 0.10)


def test_turning_rotates_the_contacts_about_the_origin():
    from weaver.interactive import apply_controls
    pts = _triple()
    out = apply_controls(pts, np.zeros(3), 0.0, np.pi / 2)
    assert np.allclose(out[0], pts[0])
    assert np.isclose(np.linalg.norm(out[1] - out[2]), 0.10)
    assert np.allclose(out[1], [-0.05, 0.0, 0.0], atol=1e-9)


def test_a_closed_agent_can_still_be_opened():
    """A fully closed gripper has no separation axis; opening must not divide by zero."""
    from weaver.interactive import apply_controls
    pts = np.zeros((3, 3))
    out = apply_controls(pts, np.zeros(3), 0.02, 0.0)
    assert np.isfinite(out).all()
    assert np.isclose(np.linalg.norm(out[1] - out[2]), 0.02)

