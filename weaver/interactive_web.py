#!/usr/bin/env python3
"""The keyboard demo, served to a browser instead of an X window.

Same thing as ``weaver.interactive`` — seed the world model with a few real
frames, then drive both agents from the keyboard and watch it imagine — but the
display is a web page, which is what makes it usable over a remote connection:
no X server on the client, and a JPEG stream instead of a raw bitmap (roughly a
twenty-fifth of the bandwidth). The model needs ~1.5 s to produce each 8-frame
chunk anyway, so 5 fps playback loses nothing.

Everything except the display is shared with the pygame version: the control
table, ``accumulate``, ``apply_controls``, ``StreamingRollout``.

No extra dependencies: the page is served by the standard library, frames go out
as ``multipart/x-mixed-replace`` (an MJPEG stream a plain ``<img>`` renders),
and the browser posts held keys back.

    .venv/bin/python -m weaver.interactive_web \
        --checkpoint runs/weaver_hri/logs/chkpts_hri \
        --overrides dataset.path=data/weaver_hri

Then open http://<host>:8420 — over Tailscale, http://100.126.38.53:8420.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from .finetune_utils import load_finetuned
from .generate_views import build_model, load_eval_config
from .interactive import (
    AGENTS, CONTROLS, Normaliser, accumulate, apply_controls, load_seed_clip,
)
from .wm.streaming import StreamingRollout

PAGE = """<!doctype html><meta charset=utf-8>
<title>WEAVER — keyboard-driven dream</title>
<style>
 body{background:#121214;color:#ddd;font:13px/1.5 ui-monospace,Menlo,monospace;
      margin:0;display:flex;flex-direction:column;align-items:center;gap:10px;padding:14px}
 img{image-rendering:pixelated;border:1px solid #333;width:__W__px;height:__H__px}
 table{border-collapse:collapse} td,th{padding:2px 9px;text-align:left}
 th{color:#888;font-weight:400} .k{color:#7cc4ff} .v{color:#eee}
 #hud{min-height:3em} #status{color:#888}
 button{background:#26262b;color:#ddd;border:1px solid #444;border-radius:4px;
        padding:4px 10px;font:inherit;cursor:pointer} button:hover{background:#34343a}
</style>
<img id=v src="/stream">
<div id=hud></div>
<div id=status>click the page, then hold keys</div>
<div><button onclick="cmd('reseed')">re-seed</button>
     <button onclick="cmd('camera')">camera</button>
     <button id=rec onclick="cmd('record')">record</button></div>
<table>
<tr><th></th><th>+X</th><th>-X</th><th>+Y</th><th>-Y</th><th>+Z</th><th>-Z</th>
    <th>open/close</th><th>turn</th></tr>
<tr><td class=k>robot</td><td>W</td><td>S</td><td>A</td><td>D</td><td>Q</td><td>E</td>
    <td>1 / 2</td><td>Z / X</td></tr>
<tr><td class=k>hand</td><td>&uarr;</td><td>&darr;</td><td>&larr;</td><td>&rarr;</td>
    <td>PgUp</td><td>PgDn</td><td>[ / ]</td><td>, / .</td></tr>
</table>
<script>
const held = new Set();
const send = () => fetch('/keys', {method:'POST', body: JSON.stringify([...held])});
addEventListener('keydown', e => {
  if (e.key === 'Tab' || e.key.startsWith('Page') || e.key.startsWith('Arrow')) e.preventDefault();
  if (!held.has(e.key)) { held.add(e.key); send(); }
});
addEventListener('keyup', e => { if (held.delete(e.key)) send(); });
addEventListener('blur', () => { held.clear(); send(); });
const cmd = a => fetch('/cmd', {method:'POST', body: a});
setInterval(async () => {
  const s = await (await fetch('/state')).json();
  document.getElementById('hud').textContent = s.hud;
  document.getElementById('status').textContent = s.status;
  document.getElementById('rec').textContent = s.recording ? 'stop recording' : 'record';
}, 500);
</script>
"""


class Demo:
    """Runs the rollout in its own thread and publishes JPEG frames."""

    def __init__(self, model, img_keys, cfg, norm, obs, seed_actions, traj_id,
                 fps: float, scale: int, record_dir: Path, cam: int = 0):
        self.model, self.img_keys, self.cfg, self.norm = model, img_keys, cfg, norm
        self.obs, self.seed_actions, self.traj_id = obs, seed_actions, traj_id
        self.fps, self.scale, self.record_dir = fps, scale, record_dir

        self.roll = StreamingRollout(model, horizon=cfg.horizon)
        self.roll.seed(obs, seed_actions)
        self.target = self.norm.denorm_state(self.roll.latest_states(1))[0, 0].cpu().numpy()
        self.lock = threading.Lock()
        self.held: set[str] = set()
        self.jpeg: bytes | None = None
        self.frame_event = threading.Event()
        self.cam = cam
        self.status = "starting"
        self.hud = ""
        self.recording: list[np.ndarray] = []
        self.is_recording = False
        self._reseed = False

    # -- state the browser pokes -------------------------------------------

    def set_keys(self, keys):
        with self.lock:
            self.held = set(keys)

    def command(self, what: str):
        if what == "camera":
            self.cam = (self.cam + 1) % len(self.img_keys)
        elif what == "reseed":
            self._reseed = True
        elif what == "record":
            self.is_recording = not self.is_recording
            if not self.is_recording and self.recording:
                self.record_dir.mkdir(parents=True, exist_ok=True)
                out = self.record_dir / f"dream_{self.traj_id}_{int(time.time())}.mp4"
                import imageio
                imageio.mimwrite(out, np.stack(self.recording), fps=self.fps)
                self.status = f"wrote {out.name} ({len(self.recording)} frames)"
                self.recording = []

    # -- the loop -----------------------------------------------------------

    def _actions(self) -> torch.Tensor:
        with self.lock:
            held = set(self.held)
        cmd = accumulate(held)
        # The action is the commanded *pose* of the three points, not a
        # displacement, so it is the target itself that goes in. With no key
        # held apply_controls returns the current points untouched, which
        # reads as "stay here" -- a zero vector would mean "go to the
        # base-frame origin".
        # The action is an absolute commanded pose, and the demo owns it: the
        # key moves self.target, which starts at the seed. Re-reading the
        # model's own predicted state each chunk would make "no key" mean "go
        # to wherever you just drifted", with nothing pulling it back. Ramped
        # across the chunk so a held key gives continuous motion.
        step = np.zeros(len(self.target))
        for agent, sl in AGENTS.items():
            pts = self.target[sl].reshape(3, 3)
            tgt = apply_controls(pts, cmd[agent]["move"], cmd[agent]["open"],
                                 cmd[agent]["turn"])
            step[sl] = (tgt - pts).reshape(9)
        n = self.roll.bootstrap
        ramp = np.arange(1, n + 1)[:, None] * step[None, :]
        chunk_cmd = self.target[None, :] + ramp
        self.target = chunk_cmd[-1].copy()
        a = torch.tensor(chunk_cmd, dtype=torch.float32,
                         device=cur_device(self.model))
        return self.norm.norm_action(a).view(1, n, -1)

    def _publish(self, frame: torch.Tensor):
        import cv2
        img = (frame.clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))                     # (H, W, C) RGB
        if self.is_recording:
            self.recording.append(img)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.scale != 1:
            bgr = cv2.resize(bgr, (bgr.shape[1] * self.scale, bgr.shape[0] * self.scale),
                             interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.jpeg = buf.tobytes()
            self.frame_event.set()

    def run(self):
        period = 1.0 / self.fps
        while True:
            if self._reseed:
                self.roll.seed(self.obs, self.seed_actions)
                self.target = self.norm.denorm_state(self.roll.latest_states(1))[0, 0].cpu().numpy()
                self._reseed = False
                self.status = "re-seeded"
            t_gen = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.roll.step(self._actions())
            gen = time.time() - t_gen

            st = self.norm.denorm_state(self.roll.latest_states(1))[0, 0].cpu().numpy()
            lines = []
            for agent, sl in AGENTS.items():
                p = st[sl].reshape(3, 3)
                lines.append(f"{agent:5s} origin {p[0][0]:+.3f} {p[0][1]:+.3f} "
                             f"{p[0][2]:+.3f}   opening "
                             f"{np.linalg.norm(p[1] - p[2]) * 100:5.1f} cm")
            self.hud = "\n".join(lines)
            self.status = (f"frame {self.roll.head}   camera {self.img_keys[self.cam]}"
                           f"   chunk {gen:.1f}s")

            frames = out[self.img_keys[self.cam]][0]
            for j in range(frames.shape[0]):
                t0 = time.time()
                self._publish(frames[j])
                time.sleep(max(0.0, period - (time.time() - t0)))


def cur_device(model) -> torch.device:
    return next(model.parameters()).device


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):           # the default logs every frame
            pass

        def _send(self, body: bytes, ctype="text/html; charset=utf-8"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                h, w = demo.cfg.dataset.image_size
                page = PAGE.replace("__W__", str(w * demo.scale)) \
                           .replace("__H__", str(h * demo.scale))
                self._send(page.encode())
            elif self.path == "/state":
                self._send(json.dumps({"hud": demo.hud, "status": demo.status,
                                       "recording": demo.is_recording}).encode(),
                           "application/json")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        demo.frame_event.wait(timeout=5.0)
                        demo.frame_event.clear()
                        jpg = demo.jpeg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode() if n else ""
            if self.path == "/keys":
                demo.set_keys(json.loads(body or "[]"))
            elif self.path == "/cmd":
                demo.command(body.strip())
            self._send(b"ok", "text/plain")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-name", default="checkpoint.pt")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--n-seed", type=int, default=None)
    ap.add_argument("--val-steps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--camera", default="exterior_1_left",
                    help="camera shown first; the rest stay one keypress away")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record-dir", type=Path, default=Path("data/interactive"))
    args = ap.parse_args()

    cfg = load_eval_config(args.checkpoint, list(args.overrides) +
                           [f"model.val_steps={args.val_steps}",
                            "inference.pyramid_stagger_width=0"])
    model, img_keys = build_model(cfg, args.device)
    load_finetuned(model, args.checkpoint, cfg,
                   checkpoint_name=args.checkpoint_name, device=args.device)
    model.eval()
    model._use_precomputed_features = True

    norm = Normaliser(cfg.dataset.norm_stats_path, args.device, cfg.dataset.path)
    n_seed = args.n_seed or max(cfg.n_history, cfg.n_memory_frames * cfg.t_memory + 1)
    obs, seed_actions, traj_id = load_seed_clip(cfg, img_keys, args.traj_index,
                                                n_seed, args.device)
    print(f"seeded from val trajectory {traj_id} with {obs['states'].shape[1]} real frames")
    print(f"cameras: {img_keys}  (showing {args.camera})")

    cam = img_keys.index(args.camera) if args.camera in img_keys else 0
    demo = Demo(model, img_keys, cfg, norm, obs, seed_actions, traj_id,
                args.fps, args.scale, args.record_dir, cam=cam)
    threading.Thread(target=demo.run, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    print(f"\n  open  http://<this-host>:{args.port}\n"
          f"  keys: {' '.join(sorted(CONTROLS))}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
