"""
Where training runs save their results. The viewer page (ui/index.html) reads them from ui/outputs/, which is
gitignored. Each run gets its own folder of JSON files:

    ui/
      index.html              the viewer: pick a run, see its replays and learning curves (tracked in git)
      outputs/
        runs.json             list of all runs, rebuilt from runs/*/run.json whenever a run is saved
        runs/<run name>/
          run.json            settings (training args, environment config) and final results
          curves.json         evaluation results over training
          replay.json         recorded episodes at each training stage (layout in runs/recording.py)

To look at them, serve the ui folder (`cd ui && python -m http.server 8000`) and open http://localhost:8000/.
Browsers don't let a page opened straight from disk load the JSON files, so opening index.html directly won't work.
"""

import json
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent.parent / "ui" / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
RUN_LIST_FIELDS = ("name", "created", "title", "summary")  # what runs.json keeps from each run.json, for the dropdown


def to_json(value, indent: int = 0, width: int = 200, expand: bool = False) -> str:
    """
    Indented JSON that keeps anything shorter than `width` on one line, so each replay frame or evaluation is one
    line. `expand` forces `value` onto several lines; maps (`tiles` lists) use it so they read as a grid.
    """
    flat = json.dumps(value)
    if not isinstance(value, (dict, list)) or (len(flat) + indent <= width and not expand):
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = [
            f"{pad}{json.dumps(key)}: {to_json(item, indent + 2, width, key == 'tiles' and isinstance(item, list))}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + to_json(item, indent + 2, width) for item in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def write_json(path: Path, data) -> Path:
    """Writes `data` to `path` with `to_json`'s layout."""
    path.write_text(to_json(data) + "\n")
    return path


def new_run_dir() -> Path:
    """
    Creates `ui/outputs/runs/<date-time>/`, e.g. ui/outputs/runs/20261101-201530/. A run's settings live in its
    run.json, not its name. Runs started in the same second (e.g. seeds launched in parallel) get -2, -3, ...
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{datetime.now():%Y%m%d-%H%M%S}"
    for attempt in range(1, 1000):
        path = RUNS_DIR / (base_name if attempt == 1 else f"{base_name}-{attempt}")
        try:
            path.mkdir()  # fails if another run already took this name, even one started at the same moment
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Couldn't find a free run folder name for {base_name}")


def register_run(run_dir: Path, run_info: dict) -> Path:
    """
    Writes the run's run.json and rebuilds ui/outputs/runs.json, the list of runs the viewer reads, from every run
    folder (so deleted runs drop out). Returns the path of runs.json.
    """
    write_json(run_dir / "run.json", {"name": run_dir.name, **run_info})
    runs = []
    for info_path in sorted(RUNS_DIR.glob("*/run.json")):
        info = json.loads(info_path.read_text())
        runs.append({key: info.get(key) for key in RUN_LIST_FIELDS})
    runs.sort(key=lambda run: run["created"] or "", reverse=True)
    return write_json(OUTPUTS_DIR / "runs.json", runs)
