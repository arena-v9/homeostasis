"""
Plotting helpers used by main.py (adapted from ARENA's `chapter2_rl/exercises/plotly_utils.py`).
"""

import numpy as np
import plotly.express as px
import plotly.io as pio
import torch as t
from IPython import get_ipython

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
