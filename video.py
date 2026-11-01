"""
Greedy rollouts and 4x4 grid videos of them (adapted from ARENA's `chapter2_rl/exercises/rl_utils.py`). Each
env draws its own cells via its `draw()` method; videos are logged to wandb if a run is active and/or shown
inline in a notebook.
"""

import numpy as np
import torch as t
from IPython.display import HTML


def _grid_frames(obs, draw_cell, dones=None, rows=4, cols=4, cell_w=160, cell_h=120, max_frames=250, flash_frames=6):
    """obs: (n_envs, T, ...) of stored observations. draw_cell(obs_vec, canvas, x0, y0, w, h) draws
    one env's cell. On each reset the cell flashes light-red for flash_frames frames."""
    import cv2

    n = rows * cols
    obs = np.asarray(obs.detach().cpu() if hasattr(obs, "detach") else obs)[:n]
    if dones is not None:
        dones = np.asarray(dones.detach().cpu() if hasattr(dones, "detach") else dones)[:n].astype(bool)
    n_have, T = obs.shape[0], obs.shape[1]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        obs = obs[:, idx]
        dones = dones[:, idx] if dones is not None else None
        T = obs.shape[1]

    red = np.zeros((n_have, T), bool)
    if dones is not None:
        for i in range(n_have):
            for ti in np.nonzero(dones[i])[0]:
                red[i, ti : ti + flash_frames] = True

    W, H = cols * cell_w, rows * cell_h
    frames = np.empty((T, H, W, 3), np.uint8)
    for ti in range(T):
        canvas = np.full((H, W, 3), 255, np.uint8)
        for i in range(n_have):
            r, c = divmod(i, cols)
            ox, oy = c * cell_w, r * cell_h
            if red[i, ti]:
                canvas[oy : oy + cell_h, ox : ox + cell_w] = (255, 200, 200)
            cv2.rectangle(canvas, (ox, oy), (ox + cell_w - 1, oy + cell_h - 1), (220, 220, 220), 1)
            draw_cell(obs[i, ti], canvas, ox, oy, cell_w, cell_h)
        frames[ti] = canvas
    return frames


def frames_to_video_html(frames, fps=50, width=640):
    """Encode (T, H, W, 3) uint8 frames as a single autoplaying/looping MP4, returned as an
    IPython HTML element (self-contained base64 data URI, so it survives notebook export)."""
    import base64
    import os
    import tempfile

    import imageio

    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    imageio.mimsave(path, list(frames), fps=float(fps))
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    os.remove(path)
    return HTML(f'<video autoplay loop controls muted src="data:video/mp4;base64,{b64}" width="{width}">')


def render_rollout_grid_html(obs, draw_fn, dones=None, fps=50, cell_w=160, cell_h=120, **kwargs):
    """Encode the first 16 stored env-trajectories as a single autoplaying/looping MP4, returned as
    an IPython HTML element. `draw_fn(obs_vec, canvas, ox, oy, w, h)` is the env's own per-cell renderer."""
    frames = _grid_frames(obs, draw_fn, dones=dones, cell_w=cell_w, cell_h=cell_h, **kwargs)
    return frames_to_video_html(frames, fps=fps)


def greedy_rollout(env, act_fn, steps=100):
    """Roll a (freshly constructed, batched) env forward `steps` steps, acting with `act_fn(obs) -> actions`
    and no exploration (for DQN, `lambda obs: q_network(obs).argmax(-1)`).
    Returns obs `(num_envs, steps, *obs_shape)`, dones `(num_envs, steps)` and each env's lifespan before
    its first termination `(num_envs,)` - used both for evaluation ("did every env survive?") and for videos."""
    obs, _ = env.reset()
    alive = t.ones(env.num_envs, dtype=t.bool, device=obs.device)
    lifespan = t.zeros(env.num_envs, device=obs.device)
    all_obs, all_dones = [], []
    with t.inference_mode():
        for _ in range(steps):
            all_obs.append(obs)
            obs, _, terminated, truncated, _ = env.step(act_fn(obs))
            all_dones.append(terminated | truncated)
            alive &= ~terminated
            lifespan += alive.float()
    return t.stack(all_obs, dim=1), t.stack(all_dones, dim=1), lifespan


class LiveVideo:
    """One display slot in a notebook cell's output. With `overwrite=True` (default) the first `show` displays
    the video with a `display_id` and every later `show` updates it in place, so the new video replaces the old
    one without wiping the progress bar above it. With `overwrite=False` every `show` appends a new video.
    Create a fresh `LiveVideo()` per training run."""

    def __init__(self, overwrite: bool = True):
        self.overwrite = overwrite
        self.handle = None

    def show(self, video, caption: str | None = None):
        from IPython.display import display

        if caption is not None:
            video = HTML(f"<div style='font-family:monospace;margin:4px 0'>{caption}</div>{video.data}")
        if self.overwrite and self.handle is not None:
            self.handle.update(video)
        else:
            self.handle = display(video, display_id=True)


def log_greedy_rollout_video(env, act_fn, draw_fn, *, key="greedy_rollout", step=None, live=False, steps=100, slot=None):
    """`greedy_rollout` followed by `log_grid_video`: roll a fresh env forward greedily and render / log /
    display the result (`slot`: a `LiveVideo` to display into). Returns the rollout's lifespans."""
    obs, dones, lifespan = greedy_rollout(env, act_fn, steps=steps)
    log_grid_video(obs, draw_fn, dones=dones, key=key, step=step, live=live, slot=slot)
    return lifespan


def log_grid_video(obs, draw_fn, dones=None, *, key="rollout_video", step=None, live=False, slot=None, caption=None, **render_kwargs):
    """Render `(obs, dones)` as a 4x4 grid MP4 with the env's own `draw_fn`, log it to wandb under `key` if a
    wandb run is active, and display it inline if `live` - into `slot` (a `LiveVideo`) if given.
    Returns the IPython HTML object, or None if rendering failed - a broken visualisation must never
    stop a training run, so errors are printed rather than raised."""
    try:
        video = render_rollout_grid_html(obs, draw_fn, dones=dones, **render_kwargs)
        return log_video_html(video, key=key, step=step, live=live, slot=slot, caption=caption)
    except Exception as e:  # never let visualization break training
        print(f"[video log skipped: {e}]")
        return None


def log_video_html(video, *, key="rollout_video", step=None, live=False, slot=None, caption=None):
    """Log an already-rendered video (an IPython HTML element) to wandb under `key` if a run is active, and
    display it inline if `live` - into `slot` (a `LiveVideo`) if given, else appended below the cell."""
    import wandb

    if wandb.run is not None:
        wandb.log({key: wandb.Html(video.data)}, step=step)
    if live:
        if slot is not None:
            slot.show(video, caption=caption)
        else:
            from IPython.display import display

            if caption is not None:
                print(caption)
            display(video)
    return video
