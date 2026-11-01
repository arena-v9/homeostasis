"""
Plotting helpers used by main.py (adapted from ARENA's `chapter2_rl/exercises/plotly_utils.py`).
"""

import einops
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch as t
from IPython import get_ipython
from plotly.subplots import make_subplots

# Outside a notebook, plotly's default "browser" renderer starts a one-shot HTTP server and blocks until a browser
# has fetched the figure - on a headless machine that never happens, so `python main.py` would hang at the first
# `fig.show()`. Fall back to the "iframe" renderer, which just writes each figure to `iframe_figures/figure_N.html`.
if get_ipython() is None and pio.renderers.default == "browser":
    pio.renderers.default = "iframe"


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def line(y: t.Tensor | list, renderer=None, **kwargs):
    """
    Line plot of `y`. Extra kwargs go to `px.line` (e.g. labels, title, height, width), except
    `return_fig=True`, which returns the figure instead of showing it.
    """
    return_fig = kwargs.pop("return_fig", False)
    fig = px.line(y=to_numpy(y), **kwargs).update_layout(hovermode="x unified")
    return fig if return_fig else fig.show(renderer=renderer)


def plot_cartpole_obs_and_dones(
    obs: np.ndarray | t.Tensor,
    done: np.ndarray | t.Tensor,
    title: str = "CartPole experiences (dotted lines = termination)",
    filename: str | None = None,
):
    """
    obs: shape (n_steps*, n_envs, obs_dim)
    dones: shape (n_steps*, n_envs)

    Plots the observations and the dones.
    """
    # Get as tensors
    if isinstance(obs, np.ndarray):
        obs = t.from_numpy(obs)
    if isinstance(done, np.ndarray):
        done = t.from_numpy(done).int()

    # done will either be (steps,) or (steps, envs), same for obs (but with an extra 4 dimension at the end)
    # Make sure they both have steps and envs dimensions concatenated (envs should be first)
    if done.ndim == 1:
        num_steps, num_envs = len(done), 1
    else:
        num_steps, num_envs = done.shape
        done = einops.rearrange(done, "steps envs -> (envs steps)", envs=num_envs)
        obs = einops.rearrange(obs, "steps envs obs -> (envs steps) obs", envs=num_envs)
    done_indices = np.nonzero(done).squeeze()

    # Make subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Cart x-position", "Pole angle"])
    fig.update_layout(
        template="simple_white",
        title=title,
        showlegend=False,
        hovermode="x unified",
        height=450,
        width=800,
        title_y=0.9,
        margin_t=120,
    )

    # Create dictionary to store the different traces (we use 6 traces: each of (max, min, actual data) for posn and angle)
    d = {"posn": obs[..., 0], "angle": obs[..., 2]}
    d["posn_min"] = np.full_like(d["posn"], -2.4)
    d["posn_max"] = np.full_like(d["posn"], +2.4)
    d["angle_min"] = np.full_like(d["posn"], -0.2095)
    d["angle_max"] = np.full_like(d["posn"], +0.2095)

    # Do the same thing for position and angle (subplots)
    for i, (name0, color) in enumerate(zip(["posn", "angle"], px.colors.qualitative.D3), 1):
        # Add traces for each of the 3 lines in this subplot: actual data, min, and max
        for name1 in ["", "_min", "_max"]:
            fig.add_trace(
                go.Scatter(y=d[name0 + name1].tolist(), name=name0 + name1, mode="lines", marker_color=color),
                col=1,
                row=i,
            )

        # Add dashed line at each termination
        for x in done_indices.tolist():
            fig.add_vline(x=x, y1=1, y0=0, line_width=2, line_color="black", line_dash="dash", col=1, row=i)

        # Add solid line at every environment jump
        if num_envs > 1:
            for x in range(num_steps, num_steps * num_envs, num_steps):
                fig.add_vline(x=x, y1=1, y0=0, line_width=4, line_color="black", col=1, row=i)

    # Add annotation for min and max
    for sign, text0 in zip([-1, 1], ["Min", "Max"]):
        for row, (y, text1) in enumerate(zip([2.4, 0.2095], ["posn", "angle"]), 1):
            fig.add_annotation(
                text=" ".join([text0, text1]),
                xref="paper",
                yref="paper",
                x=num_steps * num_envs * 1.05,
                y=sign * y,
                showarrow=False,
                row=row,
                col=1,
            )

    fig.show(config={"displaylogo": False})
    if filename is not None:
        fig.write_html(filename, config={"displaylogo": False}, include_plotlyjs="cdn")
