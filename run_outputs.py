"""
Where training runs save their results, and the viewer page for them.

Each run gets its own folder of JSON files:

    outputs/
      index.html            the viewer: pick a run, see its learning curves and replays (a copy of viewer.html)
      runs.json             list of all runs, rebuilt from runs/*/run.json whenever a run is saved
      runs/<run name>/
        run.json            settings (training args, environment config) and final results
        curves.json         evaluation results over training
        replay.json         recorded episodes at each training stage (layout in replay_viewer.py)

To look at them, serve the folder (`cd outputs && python -m http.server 8000`) and open http://localhost:8000/.
Browsers don't let a page opened straight from disk load the JSON files, so opening index.html directly won't work.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
VIEWER_PATH = Path(__file__).with_name("viewer.html")


def to_json(value, indent: int = 0, width: int = 200, expand: bool = False) -> str:
    """Indented JSON that keeps anything short enough on one line (so each replay frame or evaluation is one line).
    Maps (`tiles` lists) are always written one row per line so they read as a grid."""
    flat = json.dumps(value)
    if not isinstance(value, (dict, list)) or (len(flat) + indent <= width and not expand):
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(k)}: {to_json(v, indent + 2, width, k == 'tiles' and isinstance(v, list))}"
                 for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + to_json(v, indent + 2, width) for v in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def write_json(path: Path, data) -> Path:
    path.write_text(to_json(data) + "\n")
    return path


def new_run_dir(prefix: str) -> Path:
    """Creates `outputs/runs/<prefix>_<date-time>/`, e.g. outputs/runs/solo_seed1_200k_20261101-201530/."""
    path = RUNS_DIR / f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}"
    path.mkdir(parents=True)
    return path


def register_run(run_dir: Path, run_info: dict) -> Path:
    """Writes the run's run.json, rebuilds outputs/runs.json from every run folder (so deleted runs drop out), and
    refreshes outputs/index.html from viewer.html. Returns the viewer's path."""
    write_json(run_dir / "run.json", {"name": run_dir.name, **run_info})
    runs = []
    for info_path in sorted(RUNS_DIR.glob("*/run.json")):
        info = json.loads(info_path.read_text())
        runs.append({k: info.get(k) for k in ("name", "created", "title", "summary")})
    runs.sort(key=lambda r: r["created"] or "", reverse=True)
    write_json(OUTPUTS_DIR / "runs.json", runs)
    return Path(shutil.copyfile(VIEWER_PATH, OUTPUTS_DIR / "index.html"))
