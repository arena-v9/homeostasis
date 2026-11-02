"""
Experiment analyses for the viewer's experiment tabs (ui/index.html). An experiment compares two groups of runs, each
group being every saved run whose settings differ from the defaults in exactly the group's `config_changes`. Seeds
(training runs) are the replicates. The first group is the baseline; differences are second group minus first.

    python -m runs.experiments experiment-1

writes ui/outputs/experiments/experiment-1.json and rebuilds ui/outputs/experiments.json (the list of tabs). Re-run it
whenever runs are added.

The page is a short list of questions, each answered from the data:

  "better"          Does the second group do better? Reliability (the share of a run's evaluations after
                    RELIABLE_AFTER environment steps with survival >= RELIABLE_SURVIVAL), and deaths by starvation and
                    by fire per 1,000 test episodes over those same evaluations. Survival swings late in training
                    because policies collapse and recover; reliability measures how often that happens instead of
                    averaging it away. The learning curves show the swings directly.
  "test_dependence" Is the second group's advantage learned, or does it depend on the test setting? Every final
                    network is tested in both groups' settings (the cross-evaluation).
  "explanations"    Candidate explanations, each with its prediction and planned tests, written in the definition.

How sure we are: a 95% bootstrap interval over runs, and a two-sided permutation test (shuffling runs between
groups; for paired comparisons of the same networks, flipping the sign of each run's difference). Answer sentences
call a difference "clear" only when its 95% interval excludes zero and p < 0.05, and call a clear survival difference smaller than
NEGLIGIBLE_SURVIVAL "negligible".
"""

import argparse
import itertools
import json
import math
from datetime import datetime

import numpy as np
import torch as t

from common import device
from dqn.networks import ConvQNetwork
from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.observation import NUM_CHANNELS
from runs.outputs import OUTPUTS_DIR, RUNS_DIR, write_json
from train_gridworld import evaluate

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
CROSS_EVAL_LAYOUTS = 512
NUM_RESAMPLES = 20_000  # bootstrap resamples, and permutations when there are too many to list exactly
CROSS_EVAL_METRICS = ("survival", "deaths_starvation", "deaths_fire", "food_eaten", "return")
RELIABLE_SURVIVAL = 0.95  # an evaluation counts as "doing well" at this survival or above
RELIABLE_AFTER = 100_000  # reliability and deaths use the evaluations after this many environment steps
NEGLIGIBLE_SURVIVAL = 0.02  # clear survival differences smaller than this are called negligible

EXPERIMENTS = {
    "experiment-1": {
        "title": "Experiment 1: Does having an NPC change how the agent learns to survive?",
        "runs_per_group": 15,  # the analysis refuses to run until every group has this many runs
        "setup": (
            "Two groups of training runs that differ in one thing: whether the NPC is on the map. The NPC's health "
            "and energy are hidden, so only its presence differs. Everything else uses the default settings, and "
            "every run trains for 200k environment steps."
        ),
        "groups": [
            {"label": "No NPC", "phrase": "without the NPC", "config_changes": {"npc_enabled": False}},
            {"label": "NPC", "phrase": "with the NPC", "config_changes": {"npc_state_visible": False}},
        ],
        "questions": [
            {"kind": "better", "title": "Does an agent trained with the NPC do better?"},
            {"kind": "test_dependence",
             "title": "Is the agent learning to survive, or does it depend on the NPC being there?"},
            {"kind": "explanations", "title": "Why? Candidate explanations",
             # Written by us, not computed: edit freely
             "hypotheses": [
                 {
                     "name": "Food turnover",
                     "idea": (
                         "The NPC eats too, so food respawns in new random places more often. During training the "
                         "agent sees food in more varied positions, and learns to find it wherever it is."
                     ),
                     "prediction": (
                         "Food relocates more often per episode in NPC runs. Without the NPC, food that relocates "
                         "at the same rate should give the same advantage; an NPC that never eats should not."
                     ),
                     "tests": [
                         "Measure food relocations per episode during training, in both groups.",
                         "Train without the NPC, with food moving to a random empty square at the rate the NPC "
                         "causes (new setting).",
                         "Train with an NPC that moves but never eats (new setting).",
                     ],
                 },
                 {
                     "name": "Replay buffer: forgetting",
                     "idea": (
                         "Once an agent stops starving, starvation examples drain out of its replay buffer and it "
                         "forgets how to stay fed, so survival collapses until it starves again and relearns. The "
                         "NPC keeps creating food shortages, so starvation examples stay in the buffer."
                     ),
                     "prediction": (
                         "The gap between the groups grows with a smaller buffer (more forgetting) and shrinks to "
                         "about zero with a buffer that holds all of training (nothing is ever thrown out). Without "
                         "the NPC, the share of starvation examples in the buffer drops before each collapse."
                     ),
                     "tests": [
                         "Train both groups with 20k and 200k buffers (100k is this experiment), 15 runs each.",
                         "Log the buffer's share of low-energy, starvation and death transitions at each evaluation.",
                     ],
                 },
             ]},
        ],
        # Metrics compared in the details table (pushes and the NPC's metrics are 0 by definition without an NPC)
        "compare": ["survival", "deaths_starvation", "deaths_fire", "food_eaten", "return", "fire_steps"],
    },
}


# ---- Loading runs ---------------------------------------------------------------------------------------------------

def config_from_changes(changes: dict) -> GridworldConfig:
    """The GridworldConfig a run used, from the JSON `config_changes` saved in its run.json."""
    values = dict(changes)
    if "hazard_damage" in values:
        values["hazard_damage"] = {Tile[name.upper()]: tuple(damage) for name, damage in values["hazard_damage"].items()}
    if "start_energy_range" in values:
        values["start_energy_range"] = tuple(values["start_energy_range"])
    return GridworldConfig(**values)


def load_runs(config_changes: dict) -> list[dict]:
    """Every saved run with exactly these config changes: its run.json, curves.json and folder, sorted by seed."""
    runs = []
    for info_path in sorted(RUNS_DIR.glob("*/run.json")):
        info = json.loads(info_path.read_text())
        if info.get("config_changes") == config_changes:
            curves = json.loads((info_path.parent / "curves.json").read_text())
            runs.append({"info": info, "curves": curves, "dir": info_path.parent})
    return sorted(runs, key=lambda run: run["info"]["args"]["seed"])


# ---- Statistics -----------------------------------------------------------------------------------------------------

def compare(a, b, rng: np.random.Generator) -> dict:
    """
    Two independent groups of runs: b's mean minus a's, a 95% bootstrap interval (resampling runs within each group),
    and a two-sided permutation test (shuffling runs between the groups).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    difference = b.mean() - a.mean()
    bootstrap = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(NUM_RESAMPLES)]
    pooled, size_b = np.concatenate([a, b]), len(b)
    if math.comb(len(pooled), size_b) <= 200_000:  # few enough to list every split of runs between the groups
        splits = [list(indices) for indices in itertools.combinations(range(len(pooled)), size_b)]
        method = "exact permutation test"
    else:
        splits = [rng.permutation(len(pooled))[:size_b] for _ in range(NUM_RESAMPLES)]
        method = f"permutation test, {NUM_RESAMPLES:,} random shuffles"
    as_extreme = 0
    for indices in splits:
        in_b = np.zeros(len(pooled), dtype=bool)
        in_b[indices] = True
        as_extreme += abs(pooled[in_b].mean() - pooled[~in_b].mean()) >= abs(difference) - 1e-12
    return _result(a.mean(), b.mean(), difference, bootstrap, as_extreme / len(splits), method)


def compare_paired(a, b, rng: np.random.Generator) -> dict:
    """
    The same runs measured twice (b - a per run): the mean difference, a 95% bootstrap interval over runs, and a
    two-sided sign-flip permutation test.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    differences = b - a
    bootstrap = [rng.choice(differences, len(differences)).mean() for _ in range(NUM_RESAMPLES)]
    if len(differences) <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=len(differences))))
        method = "exact sign-flip test"
    else:
        signs = rng.choice([1, -1], size=(NUM_RESAMPLES, len(differences)))
        method = f"sign-flip test, {NUM_RESAMPLES:,} random flips"
    flipped = np.abs((signs * differences).mean(axis=1))
    p_value = float(np.mean(flipped >= abs(differences.mean()) - 1e-12))
    return _result(a.mean(), b.mean(), differences.mean(), bootstrap, p_value, method)


def _result(mean_a, mean_b, difference, bootstrap, p_value, method) -> dict:
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "a": float(mean_a), "b": float(mean_b), "difference": float(difference),
        "ci95": [float(low), float(high)], "p_value": float(p_value), "test": method,
        # Both tests must agree: the 95% interval excludes zero and the permutation test gives p < 0.05
        "clear": bool((low > 0 or high < 0) and p_value < 0.05),
    }


def probability_of_superiority(a, b) -> float:
    """How often a run picked at random from b beats one picked at random from a (ties count half)."""
    wins = sum((y > x) + 0.5 * (y == x) for x in a for y in b)
    return float(wins / (len(a) * len(b)))


# ---- Wording ---------------------------------------------------------------------------------------------------------

def format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def signed(value: float, digits: int) -> str:
    """'+0.05' or '−0.05', with no sign on a value that rounds to zero."""
    text = f"{abs(value):.{digits}f}"
    return text if float(text) == 0 else ("+" if value > 0 else "−") + text


def describe(result: dict, digits: int = 2, percent: bool = False) -> str:
    """'+0.05; 95% interval +0.02 to +0.07; p < 0.001'."""
    if percent:
        number = lambda value: f"{signed(value * 100, max(digits - 2, 0))} points"
    else:
        number = lambda value: signed(value, digits)
    low, high = result["ci95"]
    return f"{number(result['difference'])}; 95% interval {number(low)} to {number(high)}; {format_p(result['p_value'])}"


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


# ---- Analysis --------------------------------------------------------------------------------------------------------

def second_half_means(history: list[dict], metrics: list[str]) -> dict[str, float]:
    """Each metric averaged over the evaluations after half of the run's environment steps."""
    halfway = history[-1]["env_steps"] / 2
    later = [evaluation for evaluation in history if evaluation["env_steps"] > halfway]
    return {metric: float(np.mean([evaluation[metric] for evaluation in later])) for metric in metrics}


def summarize_group(runs: list[dict], metrics: list[str]) -> dict:
    """Per-run second-half averages, their summary over runs, the random policy's results and learning curves."""
    per_run = [second_half_means(run["curves"]["history"], [m for m in metrics if m in run["curves"]["metrics"]])
               for run in runs]
    group_metrics = [metric for metric in metrics if all(metric in values for values in per_run)]
    steps = [evaluation["env_steps"] for evaluation in runs[0]["curves"]["history"]]
    assert all([e["env_steps"] for e in run["curves"]["history"]] == steps for run in runs), \
        "runs in a group must be evaluated at the same environment steps"
    curves = {"env_steps": steps}
    for metric in group_metrics:
        values = np.array([[evaluation[metric] for evaluation in run["curves"]["history"]] for run in runs])
        curves[metric] = {"mean": values.mean(0).tolist(), "min": values.min(0).tolist(), "max": values.max(0).tolist()}

    def first_reaching(history, level=0.95):
        return next((e["env_steps"] for e in history if e["survival"] >= level), None)

    return {
        "runs": [
            {
                "name": run["info"]["name"],
                "seed": run["info"]["args"]["seed"],
                "second_half": values,
                "lowest_second_half_survival": min(
                    e["survival"] for e in run["curves"]["history"] if e["env_steps"] > steps[-1] / 2),
                "first_95_survival_steps": first_reaching(run["curves"]["history"]),
                "final": run["info"]["final"],
            }
            for run, values in zip(runs, per_run)
        ],
        "summary": {
            metric: {
                "mean": float(np.mean([values[metric] for values in per_run])),
                "sd": float(np.std([values[metric] for values in per_run], ddof=1)) if len(runs) > 1 else 0.0,
                "n": len(runs),
            }
            for metric in group_metrics
        },
        "random": {metric: float(np.mean([run["info"]["random"][metric] for run in runs]))
                   for metric in group_metrics if all(metric in run["info"]["random"] for run in runs)},
        "curves": curves,
    }


@t.inference_mode()
def cross_evaluate(groups: list[dict], runs_by_group: list[list[dict]]) -> list[list[list[dict]]]:
    """
    Every run's final network, tested greedily in every group's setting on CROSS_EVAL_LAYOUTS layouts.
    Returns results[trained group][tested group][run] = evaluate()'s metrics.
    """
    test_configs = [config_from_changes(group["config_changes"]) for group in groups]
    results = []
    for runs in runs_by_group:
        per_test = [[] for _ in test_configs]
        for run in runs:
            network = ConvQNetwork((NUM_CHANNELS, 7, 7), 5).to(device)
            network.load_state_dict(t.load(run["dir"] / "q_network.pt", map_location=device))
            for test_index, config in enumerate(test_configs):
                per_test[test_index].append(
                    evaluate(lambda obs: network(obs).argmax(-1), config, device, CROSS_EVAL_LAYOUTS))
        results.append(per_test)
    return results


def answer_better(groups, runs_by_group, rng) -> dict:
    """Reliability and deaths by cause per 1,000 test episodes, from each run's evaluations after RELIABLE_AFTER."""
    def per_run(history):
        later = [e for e in history if e["env_steps"] > RELIABLE_AFTER]
        return {
            "reliability": float(np.mean([e["survival"] >= RELIABLE_SURVIVAL for e in later])),
            "deaths_starvation": float(np.mean([e["deaths_starvation"] for e in later]) * 1000),
            "deaths_fire": float(np.mean([e["deaths_fire"] for e in later]) * 1000),
            "evaluations": len(later),
        }
    values = [[{**per_run(run["curves"]["history"]), "seed": run["info"]["args"]["seed"], "name": run["info"]["name"]}
               for run in group_runs] for group_runs in runs_by_group]
    measures = {
        "reliability": ("reliability", True),  # (name, higher is better)
        "deaths_starvation": ("starvation deaths", False),
        "deaths_fire": ("fire deaths", False),
    }
    results = {key: compare([v[key] for v in values[0]], [v[key] for v in values[1]], rng) for key in measures}
    better, worse, unclear = [], [], []
    for key, (name, higher_is_better) in measures.items():
        result = results[key]
        if not result["clear"]:
            unclear.append(name)
        elif (result["difference"] > 0) == higher_is_better:
            better.append(name)
        else:
            worse.append(name)
    join = lambda names: ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
    parts = []
    if better:
        parts.append(f"better on {join(better)}")
    if worse:
        parts.append(f"worse on {join(worse)}")
    if unclear:
        parts.append(f"no clear difference in {join(unclear)}")
    verdict = "; ".join(parts)
    verdict = verdict[0].upper() + verdict[1:] + "."
    base, other = groups[0]["phrase"], groups[1]["phrase"]
    reliability, starvation, fire = results["reliability"], results["deaths_starvation"], results["deaths_fire"]
    answer = (
        f"{verdict} Reliability (share of evaluations after {RELIABLE_AFTER // 1000}k steps at "
        f"{RELIABLE_SURVIVAL:.0%} survival or more): {reliability['b']:.0%} trained {other}, {reliability['a']:.0%} "
        f"trained {base} (difference {signed(reliability['difference'] * 100, 0)} points, 95% interval "
        f"{signed(reliability['ci95'][0] * 100, 0)} to {signed(reliability['ci95'][1] * 100, 0)}, "
        f"{format_p(reliability['p_value'])}). Starvation deaths per 1,000 test episodes: {starvation['b']:.0f} "
        f"against {starvation['a']:.0f} ({describe(starvation, digits=0)}). Fire deaths per 1,000: "
        f"{fire['b']:.1f} against {fire['a']:.1f} ({describe(fire, digits=1)})."
    )
    short = (f"{verdict} Reliability {reliability['b']:.0%} against {reliability['a']:.0%}; starvation deaths "
             f"{starvation['b']:.0f} against {starvation['a']:.0f} per 1,000 episodes; fire deaths "
             f"{fire['b']:.1f} against {fire['a']:.1f}.")
    return {
        "answer": answer,
        "short": short,
        "results": results,
        "per_run": values,
        "threshold": RELIABLE_SURVIVAL,
        "after_steps": RELIABLE_AFTER,
        "evaluations_per_run": values[0][0]["evaluations"],
        "probability_of_superiority": probability_of_superiority(
            [v["reliability"] for v in values[0]], [v["reliability"] for v in values[1]]),
    }


def answer_explanations(question) -> dict:
    names = [hypothesis["name"].split(":")[0].lower() for hypothesis in question["hypotheses"]]
    short = f"Not tested yet. Candidates: {' and '.join(names)}."
    return {"answer": short, "short": short}


def answer_test_dependence(groups, cross, rng) -> dict:
    """Is group 1's advantage learned (6a) and does it depend on the test setting (6b)?"""
    survival = lambda trained, tested: [r["survival"] for r in cross[trained][tested]]
    # Same test (the baseline setting), different training
    same_test = compare(survival(0, 0), survival(1, 0), rng)
    # Same networks (group 1's), tested in the baseline setting vs their own
    removed = compare_paired(survival(1, 1), survival(1, 0), rng)
    grid = [[{"mean": float(np.mean(survival(trained, tested))), "sd": float(np.std(survival(trained, tested), ddof=1)),
              "per_run": survival(trained, tested)}
             for tested in range(len(groups))] for trained in range(len(groups))]
    base, other = groups[0]["phrase"], groups[1]["phrase"]
    meaningful = lambda result: result["clear"] and abs(result["difference"]) >= NEGLIGIBLE_SURVIVAL
    if meaningful(removed):
        verdict = "It partly depends on the NPC being there."
    elif meaningful(same_test) and same_test["difference"] > 0:
        verdict = "It is learning to survive."
    else:
        verdict = "No sign that it depends on the NPC being there, but the final networks alone can't confirm it."
    note = lambda result: (f"; consistent but negligible, under {NEGLIGIBLE_SURVIVAL}"
                           if result["clear"] and not meaningful(result) else "")
    answer = (f"{verdict} Agents trained {other}, tested {base} instead: {removed['b']:.2f}, against "
              f"{removed['a']:.2f} tested as trained (difference {describe(removed)}{note(removed)}). Tested in the "
              f"same setting "
              f"({base}), agents trained {other}: {same_test['b']:.2f}; trained {base}: {same_test['a']:.2f} "
              f"(difference {describe(same_test)}).")
    return {"answer": answer, "short": verdict, "same_test": same_test, "removed": removed, "grid": grid,
            "layouts": CROSS_EVAL_LAYOUTS}


def analyze(experiment_id: str, allow_incomplete: bool = False) -> dict:
    experiment = EXPERIMENTS[experiment_id]
    groups = experiment["groups"]
    assert len(groups) == 2, "the analysis compares exactly two groups"
    runs_by_group = [load_runs(group["config_changes"]) for group in groups]
    counts = {group["label"]: len(runs) for group, runs in zip(groups, runs_by_group)}
    if not allow_incomplete and any(count != experiment["runs_per_group"] for count in counts.values()):
        raise SystemExit(f"expected {experiment['runs_per_group']} runs per group, found {counts}; train the missing "
                         "runs first (or pass --allow-incomplete)")
    metric_labels = {}
    for runs in runs_by_group:
        for run in runs:
            metric_labels.update(run["curves"]["metrics"])
    summaries = [summarize_group(runs, list(metric_labels)) for runs in runs_by_group]
    cross = cross_evaluate(groups, runs_by_group)
    rng = np.random.default_rng(0)

    answers = []
    for question in experiment["questions"]:
        if question["kind"] == "better":
            answer = answer_better(groups, runs_by_group, rng)
        elif question["kind"] == "explanations":
            answer = answer_explanations(question)
        elif question["kind"] == "test_dependence":
            answer = answer_test_dependence(groups, cross, rng)
        else:
            raise ValueError(f"unknown question kind {question['kind']!r}")
        answers.append({**question, **answer})

    compared = [m for m in experiment["compare"] if all(m in summary["summary"] for summary in summaries)]
    details = {
        "comparison": {metric: compare([run["second_half"][metric] for run in summaries[0]["runs"]],
                                       [run["second_half"][metric] for run in summaries[1]["runs"]], rng)
                       for metric in compared},
        "learning_speed": compare(*[[run["first_95_survival_steps"] or math.inf for run in summary["runs"]]
                                    for summary in summaries], rng)
        if all(run["first_95_survival_steps"] for summary in summaries for run in summary["runs"]) else None,
        "stability": compare(*[[run["lowest_second_half_survival"] for run in summary["runs"]]
                               for summary in summaries], rng),
    }
    return {
        "id": experiment_id,
        "title": experiment["title"],
        "setup": experiment["setup"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "metrics": metric_labels,
        "groups": [{**group, **summary} for group, summary in zip(groups, summaries)],
        "questions": answers,
        "details": details,
    }


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
