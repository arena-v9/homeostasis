"""
Analyses an experiment (see runs/experiment_definitions.py) from its saved runs, for the viewer's experiment tab:

    python -m runs.experiments experiment-1

writes ui/outputs/experiments/experiment-1.json and rebuilds ui/outputs/experiments.json, the list of tabs. Re-run it
whenever runs are added.

Survival swings late in training, as agents forget and relearn how to stay fed, so each run is summarised by how often
it does well rather than by an average: its reliability is the share of evaluations after RELIABLE_AFTER environment
steps with at least RELIABLE_SURVIVAL survival. Answers call a difference "clear" as defined in runs/statistics.py,
and a clear survival difference smaller than NEGLIGIBLE_SURVIVAL "negligible".
"""

import argparse
import json
from datetime import datetime

import numpy as np
import torch as t

from common import device
from dqn.networks import ConvQNetwork
from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import Action
from envs.gridworld.episode import observation_shape
from runs.evaluation import evaluate
from runs.experiment_definitions import EXPERIMENTS
from runs.outputs import OUTPUTS_DIR, RUNS_DIR, write_json
from runs.statistics import compare, compare_paired

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
RELIABLE_SURVIVAL = 0.95  # an evaluation counts as doing well at this survival or above
RELIABLE_AFTER = 100_000  # reliability and deaths use the evaluations after this many environment steps
NEGLIGIBLE_SURVIVAL = 0.02
CROSS_EVAL_LAYOUTS = 512  # layouts each final network is tested on, in each group's setting


def analyze(experiment_id: str, allow_incomplete: bool = False) -> dict:
    experiment = EXPERIMENTS[experiment_id]
    groups = experiment["groups"]
    runs_by_group = [load_runs(group) for group in groups]
    counts = {group["label"]: len(runs) for group, runs in zip(groups, runs_by_group)}
    if not allow_incomplete and any(count != experiment["runs_per_group"] for count in counts.values()):
        raise SystemExit(f"expected {experiment['runs_per_group']} runs per group, found {counts}; train the missing "
                         "runs first (or pass --allow-incomplete)")

    rng = np.random.default_rng(0)
    answer = {
        "better": lambda question: answer_better(groups, runs_by_group, rng),
        "test_dependence": lambda question: answer_test_dependence(groups, runs_by_group, rng),
        "explanations": answer_explanations,
    }
    return {
        "id": experiment_id,
        "title": experiment["title"],
        "setup": experiment["setup"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "metrics": curve_labels(runs_by_group[0][0]),
        "groups": [{**group, **summarize(runs)} for group, runs in zip(groups, runs_by_group)],
        "questions": [{**question, **answer[question["kind"]](question)} for question in experiment["questions"]],
    }


# ---- Runs -----------------------------------------------------------------------------------------------------------

def load_runs(group: dict) -> list[dict]:
    """The group's saved runs (run.json, curves.json and folder), sorted by seed."""
    runs = []
    for info_path in sorted(RUNS_DIR.glob("*/run.json")):
        info = json.loads(info_path.read_text())
        if info["config_changes"] == group["config_changes"]:
            curves = json.loads((info_path.parent / "curves.json").read_text())
            runs.append({"info": info, "curves": curves, "dir": info_path.parent})
    return sorted(runs, key=lambda run: run["info"]["args"]["seed"])


def curve_labels(run: dict) -> dict[str, str]:
    """Everything recorded at each evaluation: the evaluation results, then the forgetting diagnostics."""
    return {**run["curves"]["metrics"], **run["curves"]["diagnostics"]}


def summarize(runs: list[dict]) -> dict:
    """What the page shows for a group: its runs, the random policy's results and the learning curves over runs."""
    metrics = curve_labels(runs[0])
    steps = [evaluation["env_steps"] for evaluation in runs[0]["curves"]["history"]]
    curves = {"env_steps": steps}
    for metric in metrics:
        values = np.array([[evaluation[metric] for evaluation in run["curves"]["history"]] for run in runs])
        curves[metric] = {"mean": values.mean(0).tolist(), "min": values.min(0).tolist(), "max": values.max(0).tolist()}
    return {
        "runs": [{"name": run["info"]["name"], "seed": run["info"]["args"]["seed"]} for run in runs],
        "random": {metric: float(np.mean([run["info"]["random"][metric] for run in runs]))
                   for metric in runs[0]["curves"]["metrics"]},  # diagnostics have no random-policy value
        "curves": curves,
    }


def load_network(run: dict, config: GridworldConfig) -> ConvQNetwork:
    network = ConvQNetwork(observation_shape(config), len(Action)).to(device)
    network.load_state_dict(t.load(run["dir"] / "q_network.pt", map_location=device))
    return network


# ---- Answers --------------------------------------------------------------------------------------------------------

def answer_better(groups: list[dict], runs_by_group: list[list[dict]], rng: np.random.Generator) -> dict:
    """Reliability, and deaths by cause per 1,000 test episodes, over each run's evaluations after RELIABLE_AFTER."""
    per_run = [[run_measures(run) for run in runs] for runs in runs_by_group]
    results = {
        measure: compare([run[measure] for run in per_run[0]], [run[measure] for run in per_run[1]], rng)
        for measure in ("reliability", "deaths_starvation", "deaths_fire")
    }
    reliability, starvation, fire = results["reliability"], results["deaths_starvation"], results["deaths_fire"]
    verdict = better_verdict(results)
    base, other = groups[0]["phrase"], groups[1]["phrase"]
    return {
        "short": (f"{verdict} Reliability {reliability['b']:.0%} against {reliability['a']:.0%}; starvation deaths "
                  f"{starvation['b']:.0f} against {starvation['a']:.0f} per 1,000 episodes; fire deaths "
                  f"{fire['b']:.1f} against {fire['a']:.1f}."),
        "answer": (f"{verdict} Reliability: {reliability['b']:.0%} trained {other}, {reliability['a']:.0%} trained "
                   f"{base} ({describe(reliability, digits=0, scale=100, unit=' points')}). Starvation deaths per "
                   f"1,000 test episodes: {starvation['b']:.0f} against {starvation['a']:.0f} "
                   f"({describe(starvation, digits=0)}). Fire deaths per 1,000: {fire['b']:.1f} against "
                   f"{fire['a']:.1f} ({describe(fire, digits=1)})."),
        "results": results,
        "per_run": per_run,
        "threshold": RELIABLE_SURVIVAL,
        "evaluations_per_run": per_run[0][0]["evaluations"],
    }


def run_measures(run: dict) -> dict:
    later = [evaluation for evaluation in run["curves"]["history"] if evaluation["env_steps"] > RELIABLE_AFTER]
    return {
        "name": run["info"]["name"],
        "seed": run["info"]["args"]["seed"],
        "reliability": float(np.mean([evaluation["survival"] >= RELIABLE_SURVIVAL for evaluation in later])),
        "deaths_starvation": float(np.mean([evaluation["deaths_starvation"] for evaluation in later]) * 1000),
        "deaths_fire": float(np.mean([evaluation["deaths_fire"] for evaluation in later]) * 1000),
        "evaluations": len(later),
    }


def better_verdict(results: dict) -> str:
    """E.g. "Better on reliability and starvation deaths; no clear difference in fire deaths." """
    names = {"reliability": "reliability", "deaths_starvation": "starvation deaths", "deaths_fire": "fire deaths"}
    better, worse, unclear = [], [], []
    for measure, result in results.items():
        higher_is_better = measure == "reliability"
        if not result["clear"]:
            unclear.append(names[measure])
        elif (result["difference"] > 0) == higher_is_better:
            better.append(names[measure])
        else:
            worse.append(names[measure])
    parts = [f"{prefix} {' and '.join(items)}"
             for prefix, items in (("better on", better), ("worse on", worse), ("no clear difference in", unclear))
             if items]
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


@t.inference_mode()
def answer_test_dependence(groups: list[dict], runs_by_group: list[list[dict]], rng: np.random.Generator) -> dict:
    """
    Tests every final network in both groups' settings. Two comparisons: the second group's networks without vs with
    their own setting (does the advantage depend on it?), and both groups' networks in the baseline setting (is the
    advantage still there when the test is identical?).
    """
    configs = [GridworldConfig(**group["config_changes"]) for group in groups]
    survival = [[[] for _ in configs] for _ in configs]  # survival[trained group][tested group][run]
    for trained, runs in enumerate(runs_by_group):
        for run in runs:
            network = load_network(run, configs[trained])
            for tested, config in enumerate(configs):
                result = evaluate(lambda obs: network(obs).argmax(-1), config, device, CROSS_EVAL_LAYOUTS)
                survival[trained][tested].append(result["survival"])
    removed = compare_paired(survival[1][1], survival[1][0], rng)
    same_test = compare(survival[0][0], survival[1][0], rng)

    meaningful = lambda result: result["clear"] and abs(result["difference"]) >= NEGLIGIBLE_SURVIVAL
    base, other = groups[0]["phrase"], groups[1]["phrase"]
    if meaningful(removed):
        verdict = "It partly depends on the NPC being there."
    elif meaningful(same_test) and same_test["difference"] > 0:
        verdict = "It is learning to survive."
    else:
        verdict = "No sign that it depends on the NPC being there, but the final networks alone can't confirm it."
    negligible = ""
    if removed["clear"] and not meaningful(removed):
        negligible = f"; consistent but negligible, under {NEGLIGIBLE_SURVIVAL}"
    return {
        "short": verdict,
        "answer": (f"{verdict} Agents trained {other}, tested {base} instead: {removed['b']:.2f}, against "
                   f"{removed['a']:.2f} tested as trained ({describe(removed)}{negligible}). Tested in the same "
                   f"setting ({base}): {same_test['b']:.2f} trained {other}, {same_test['a']:.2f} trained {base} "
                   f"({describe(same_test)})."),
        "grid": [[{"mean": float(np.mean(cell)), "sd": float(np.std(cell, ddof=1))} for cell in row]
                 for row in survival],
        "layouts": CROSS_EVAL_LAYOUTS,
    }


def answer_explanations(question: dict) -> dict:
    names = " and ".join(hypothesis["name"].split(":")[0].lower() for hypothesis in question["hypotheses"])
    short = f"Not tested yet. Candidates: {names}."
    return {"short": short, "answer": short}


def describe(result: dict, digits: int = 2, scale: float = 1, unit: str = "") -> str:
    """A difference with its interval and p-value: "+0.05; 95% interval +0.02 to +0.07; p < 0.001"."""
    def signed(value: float) -> str:
        text = f"{abs(value * scale):.{digits}f}"
        return text if float(text) == 0 else ("+" if value > 0 else "−") + text
    low, high = result["ci95"]
    p_value = "p < 0.001" if result["p_value"] < 0.001 else f"p = {result['p_value']:.3f}"
    return f"{signed(result['difference'])}{unit}; 95% interval {signed(low)} to {signed(high)}; {p_value}"


# ---- Output ---------------------------------------------------------------------------------------------------------

def write_experiment(result: dict) -> None:
    """Saves one experiment's analysis and rebuilds experiments.json, the list of experiment tabs."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EXPERIMENTS_DIR / f"{result['id']}.json", result)
    index = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        saved = json.loads(path.read_text())
        index.append({"id": saved["id"], "title": saved["title"], "created": saved["created"]})
    write_json(OUTPUTS_DIR / "experiments.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--allow-incomplete", action="store_true", help="analyse even if groups are missing runs")
    cli = parser.parse_args()
    result = analyze(cli.experiment, cli.allow_incomplete)
    write_experiment(result)
    for number, question in enumerate(result["questions"], 1):
        print(f"{number}. {question['title']}\n   {question['answer']}\n")
    print(f"saved {EXPERIMENTS_DIR / (result['id'] + '.json')}")


if __name__ == "__main__":
    main()
