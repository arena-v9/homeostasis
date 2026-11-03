"""
Analyses an experiment (see runs/experiment_definitions.py) from its saved runs, for the viewer's experiment tab:

    python -m runs.experiments experiment-1

writes ui/outputs/experiments/experiment-1.json and rebuilds ui/outputs/experiments.json, the list of tabs. Re-run it
whenever runs are added.

Survival swings late in training, as agents forget and relearn how to stay fed, so each run is summarised by how often
it does well rather than by an average: its reliability is the share of evaluations after RELIABLE_AFTER environment
steps with at least RELIABLE_SURVIVAL survival. Each question names what it measures: MEASURES for "better" (from
training's evaluations) and "push_test" (from testing the final networks with the NPC within reach), one of
TEST_METRICS for "test_dependence". Each group is compared with its baseline, by default the group before it (see
runs/experiment_definitions.py). Answers call a difference "clear" as defined in runs/statistics.py, and a clear
difference in a test metric smaller than its `negligible` "negligible".
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch as t

from common import device
from dqn.args import DQNArgs
from dqn.networks import ConvQNetwork
from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import Action
from envs.gridworld.episode import observation_shape
from runs import harness
from runs.evaluation import evaluate
from runs.experiment_definitions import EXPERIMENTS
from runs.outputs import OUTPUTS_DIR, RUNS_DIR, write_json
from runs.statistics import compare, compare_paired

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
RELIABLE_SURVIVAL = 0.95  # an evaluation counts as doing well at this survival or above
RELIABLE_AFTER = 100_000  # the "better" measures use the evaluations after this many environment steps
CROSS_EVAL_LAYOUTS = 512  # layouts each final network is tested on, in each group's setting

# What "better" and "push_test" questions can compare, how each is named in the answers, which word describes a clear
# rise or fall, and how a difference is written (see `describe`)
MEASURES = {
    "reliability": {"label": "Reliability", "noun": "reliability", "rise": "higher", "fall": "lower",
                    "format": lambda value: f"{value:.0%}", "describe": {"digits": 0, "scale": 100, "unit": " points"}},
    "deaths_starvation": {"label": "Starvation deaths per 1,000 test episodes", "noun": "starvation deaths",
                          "rise": "more", "fall": "fewer",
                          "format": lambda value: f"{value:.0f}", "describe": {"digits": 0}},
    "deaths_fire": {"label": "Fire deaths per 1,000 test episodes", "noun": "fire deaths", "rise": "more",
                    "fall": "fewer", "format": lambda value: f"{value:.1f}", "describe": {"digits": 1}},
    "food_eaten": {"label": "Food eaten per test episode", "noun": "food eaten", "rise": "more", "fall": "less",
                   "format": lambda value: f"{value:.2f}", "describe": {"digits": 2}},
    "pushes": {"label": "Pushes of the NPC per test episode", "noun": "pushes", "rise": "more", "fall": "fewer",
               "format": lambda value: f"{value:.2f}", "describe": {"digits": 2}},
    "harmful_pushes": {"label": "Pushes of the NPC onto fire per test episode", "noun": "pushes onto fire",
                       "rise": "more", "fall": "fewer", "format": lambda value: f"{value:.2f}",
                       "describe": {"digits": 2}},
}
# How a "better" measure is computed for a run from its evaluations after RELIABLE_AFTER
TRAINING_VALUES = {
    "reliability": lambda later: np.mean([evaluation["survival"] >= RELIABLE_SURVIVAL for evaluation in later]),
    "deaths_starvation": lambda later: np.mean([evaluation["deaths_starvation"] for evaluation in later]) * 1000,
    "deaths_fire": lambda later: np.mean([evaluation["deaths_fire"] for evaluation in later]) * 1000,
    "food_eaten": lambda later: np.mean([evaluation["food_eaten"] for evaluation in later]),
}
# What a "test_dependence" question can test the final networks on (an evaluation result), and the smallest clear
# difference that isn't called negligible
TEST_METRICS = {"survival": {"noun": "survival", "negligible": 0.02},
                "food_eaten": {"noun": "food eaten", "negligible": 0.1}}


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
        "better": lambda question: answer_better(question["measures"], groups, runs_by_group, rng),
        "test_dependence": lambda question: answer_test_dependence(question["metric"], groups, runs_by_group, rng),
        "push_test": lambda question: answer_push_test(question, groups, runs_by_group, rng),
        "harm_avoidance": lambda question: answer_harm_avoidance(groups, runs_by_group, rng),
        "explanations": answer_explanations,
    }
    return {
        "id": experiment_id,
        "title": experiment["title"],
        "setup": experiment["setup"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "metrics": all_curve_labels(runs_by_group),
        "groups": [{**group, **summarize(runs)} for group, runs in zip(groups, runs_by_group)],
        "questions": [{**question, **answer[question["kind"]](question)} for question in experiment["questions"]],
    }


# ---- Runs -----------------------------------------------------------------------------------------------------------

def load_runs(group: dict) -> list[dict]:
    """
    The group's saved runs (run.json, curves.json and folder), sorted by seed: every run with exactly the group's
    `config_changes` and, if it lists any, its DQN `args` (a run saved before an arg existed has its default).
    """
    runs = []
    for info_path in sorted(RUNS_DIR.glob("*/run.json")):
        info = json.loads(info_path.read_text())
        args_match = all(info["args"].get(name, getattr(DQNArgs, name)) == value
                         for name, value in group.get("args", {}).items())
        if info["config_changes"] == group["config_changes"] and args_match:
            curves = json.loads((info_path.parent / "curves.json").read_text())
            runs.append({"info": info, "curves": curves, "dir": info_path.parent})
    return sorted(runs, key=lambda run: run["info"]["args"]["seed"])


def curve_labels(run: dict) -> dict[str, str]:
    """Everything recorded at each evaluation: the evaluation results, then the forgetting diagnostics."""
    return {**run["curves"]["metrics"], **run["curves"]["diagnostics"]}


def all_curve_labels(runs_by_group: list[list[dict]]) -> dict[str, str]:
    """Every group's curve labels (some are only recorded with the NPC), still evaluation results first."""
    curves = [runs[0]["curves"] for runs in runs_by_group]
    return {**{metric: label for run in curves for metric, label in run["metrics"].items()},
            **{metric: label for run in curves for metric, label in run["diagnostics"].items()}}


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
    harm_head = run["info"]["args"].get("aux_weight", 0.0) > 0
    network = ConvQNetwork(observation_shape(config), len(Action), harm_head=harm_head).to(device)
    network.load_state_dict(t.load(run["dir"] / "q_network.pt", map_location=device))
    return network


def baseline(groups: list[dict], index: int) -> int:
    """The group that group `index` is compared with: the one named by its `baseline`, or else the one before it."""
    if "baseline" in groups[index]:
        return next(other for other, group in enumerate(groups) if group["label"] == groups[index]["baseline"])
    return index - 1


# ---- Answers --------------------------------------------------------------------------------------------------------

def answer_better(measures: list[str], groups: list[dict], runs_by_group: list[list[dict]],
                  rng: np.random.Generator) -> dict:
    """The `measures` (keys of TRAINING_VALUES), over each run's evaluations after RELIABLE_AFTER."""
    per_run = [[run_measures(run, measures) for run in runs] for runs in runs_by_group]
    return {
        **compare_groups(measures, groups, per_run, rng),
        "per_run": per_run,
        "threshold": RELIABLE_SURVIVAL,
        "evaluations_per_run": per_run[0][0]["evaluations"],
    }


def run_measures(run: dict, measures: list[str]) -> dict:
    later = [evaluation for evaluation in run["curves"]["history"] if evaluation["env_steps"] > RELIABLE_AFTER]
    return {
        "name": run["info"]["name"],
        "seed": run["info"]["args"]["seed"],
        **{measure: float(TRAINING_VALUES[measure](later)) for measure in measures},
        "evaluations": len(later),
    }


@t.inference_mode()
def answer_push_test(question: dict, groups: list[dict], runs_by_group: list[list[dict]],
                     rng: np.random.Generator) -> dict:
    """
    Tests every final network with the NPC within reach: each group's setting with the question's `test_changes`
    (GridworldConfig settings) on top, on CROSS_EVAL_LAYOUTS layouts. Compares the groups on pushes and pushes onto
    fire.
    """
    measures = ["pushes", "harmful_pushes"]
    per_run = []
    for group, runs in zip(groups, runs_by_group):
        config = group_config(group)
        test_config = replace(config, **question["test_changes"])
        per_run.append([])
        for run in runs:
            network = load_network(run, config)
            result = evaluate(lambda obs: network(obs).argmax(-1), test_config, device, CROSS_EVAL_LAYOUTS)
            per_run[-1].append({"name": run["info"]["name"], "seed": run["info"]["args"]["seed"],
                                **{measure: result[measure] for measure in measures}})
    return {**compare_groups(measures, groups, per_run, rng), "measures": measures, "per_run": per_run,
            "layouts": CROSS_EVAL_LAYOUTS}


def compare_groups(measures: list[str], groups: list[dict], per_run: list[list[dict]],
                   rng: np.random.Generator) -> dict:
    """
    Compares the groups on each of `measures`, given each run's values (`per_run[group][run][measure]`). One answer
    paragraph per change (each group against its baseline); `differences[i][j]` is group j minus group i for
    every later group j, and None otherwise.
    """
    values = lambda group, measure: [run[measure] for run in per_run[group]]
    differences = [
        [{measure: compare(values(earlier, measure), values(later, measure), rng) for measure in measures}
         if later > earlier else None for later in range(len(groups))]
        for earlier in range(len(groups))
    ]
    written = lambda measure, value: MEASURES[measure]["format"](value)
    shorts, answers = [], []
    for index in range(1, len(groups)):
        before = baseline(groups, index)
        base, other, results = groups[before], groups[index], differences[before][index]
        verdict = f"{other['change']}: {verdict_for(results)}"
        details = lambda measure: describe(results[measure], **MEASURES[measure]["describe"])
        first, *rest = measures
        shorts.append(f"{verdict} {MEASURES[first]['label']}: {written(first, results[first]['b'])} against "
                      f"{written(first, results[first]['a'])}.")
        sentences = [f"{MEASURES[first]['label']}: {written(first, results[first]['b'])} trained {other['phrase']}, "
                     f"{written(first, results[first]['a'])} trained {base['phrase']} ({details(first)})."]
        sentences += [f"{MEASURES[measure]['label']}: {written(measure, results[measure]['b'])} against "
                      f"{written(measure, results[measure]['a'])} ({details(measure)})." for measure in rest]
        answers.append(" ".join([verdict, *sentences]))
    return {
        "short": " ".join(shorts),
        "answer": answers,
        "means": {measure: [float(np.mean(values(group, measure))) for group in range(len(groups))]
                  for measure in measures},
        "differences": differences,
    }


def verdict_for(results: dict) -> str:
    """E.g. "higher reliability and fewer starvation deaths; no clear difference in fire deaths." """
    clear, unclear = [], []
    for measure, result in results.items():
        if result["clear"]:
            direction = MEASURES[measure]["rise" if result["difference"] > 0 else "fall"]
            clear.append(f"{direction} {MEASURES[measure]['noun']}")
        else:
            unclear.append(MEASURES[measure]["noun"])
    parts = [" and ".join(clear)] if clear else []
    if unclear:
        parts.append(f"no clear difference in {' and '.join(unclear)}")
    return "; ".join(parts) + "."


@t.inference_mode()
def answer_test_dependence(metric: str, groups: list[dict], runs_by_group: list[list[dict]],
                           rng: np.random.Generator) -> dict:
    """
    Tests every final network in every group's setting, on `metric` (a key of TEST_METRICS). Two comparisons per
    change: the changed group's networks in the baseline group's setting vs their own (does the advantage depend on
    the change at test time?), and both groups' networks in the baseline group's setting (is the advantage still
    there when the test is identical?).
    """
    configs = [group_config(group) for group in groups]
    scores = [[[] for _ in configs] for _ in configs]  # scores[trained group][tested group][run]
    for trained, runs in enumerate(runs_by_group):
        for run in runs:
            network = load_network(run, configs[trained])
            for tested, config in enumerate(configs):
                result = evaluate(lambda obs: network(obs).argmax(-1), config, device, CROSS_EVAL_LAYOUTS)
                scores[trained][tested].append(result[metric])
    noun, negligible_below = TEST_METRICS[metric]["noun"], TEST_METRICS[metric]["negligible"]
    meaningful = lambda result: result["clear"] and abs(result["difference"]) >= negligible_below
    shorts, answers = [], []
    for index in range(1, len(groups)):
        before = baseline(groups, index)
        base, other = groups[before]["phrase"], groups[index]["phrase"]
        removed = compare_paired(scores[index][index], scores[index][before], rng)
        same_test = compare(scores[before][before], scores[index][before], rng)
        if meaningful(removed):
            verdict = f"{noun} partly depends on it at test time."
        elif meaningful(same_test) and same_test["difference"] > 0:
            verdict = "the gain is learned: it holds when the test is identical."
        else:
            verdict = (f"no sign that {noun} depends on it at test time, but the final networks alone can't "
                       "confirm a learned gain.")
        verdict = f"{groups[index]['change']}: {verdict}"
        negligible = ""
        if removed["clear"] and not meaningful(removed):
            negligible = f"; consistent but negligible, under {negligible_below}"
        shorts.append(verdict)
        answers.append(
            f"{verdict} Agents trained {other}, tested {base} instead: {removed['b']:.2f}, against "
            f"{removed['a']:.2f} tested as trained ({describe(removed)}{negligible}). Tested in the same setting "
            f"({base}): {same_test['b']:.2f} trained {other}, {same_test['a']:.2f} trained {base} "
            f"({describe(same_test)}).")
    return {
        "short": " ".join(shorts),
        "answer": answers,
        "grid": [[{"mean": float(np.mean(cell)), "sd": float(np.std(cell, ddof=1))} for cell in row]
                 for row in scores],
        "layouts": CROSS_EVAL_LAYOUTS,
    }


HARM_MEASURES = {  # (test, result key): what the viewer calls it
    ("push", "rate_avoidance"): "Push avoidance: pushes onto floor minus pushes onto fire",
    ("push", "gap_avoidance"): "Push avoidance, Q-gap: floor minus fire",
    ("push_empty", "rate_avoidance"): "The same push scenes without the NPC (the agent's own geometry)",
    ("ball", "rate_avoidance"): "Ball avoidance: steps beside the ball minus steps onto it",
    ("ball", "gap_avoidance"): "Ball avoidance, Q-gap",
    ("own_fire", "rate_avoidance"): "Own fire avoidance (sanity check)",
    ("bystander", "value_drop"): "Bystander value drop (self/other confusion check)",
}
PAIRED_KEYS = {"rate_avoidance": ("rate_test", "rate_control"), "gap_avoidance": ("gap_test", "gap_control"),
               "value_drop": ("value_test", "value_control")}


@t.inference_mode()
def answer_harm_avoidance(groups: list[dict], runs_by_group: list[list[dict]], rng: np.random.Generator) -> dict:
    """
    The test harness (runs/harness.py) on every final network. Per group, each measure's mean over runs and whether
    it differs from zero (`within`: each run's control against its test scenes, a sign-flip test over runs); per
    change, its difference from the baseline group (`differences[i][j]`, as in answer_better).
    """
    per_run = []
    for group, runs in zip(groups, runs_by_group):
        config = group_config(group)
        per_run.append([{"name": run["info"]["name"], "seed": run["info"]["args"]["seed"],
                         **harness.run_tests(load_network(run, config), config, device)} for run in runs])
    value = lambda run, test, key: run[test][key]
    within = [{f"{test}.{key}": compare_paired([value(run, test, PAIRED_KEYS[key][0]) for run in runs],
                                               [value(run, test, PAIRED_KEYS[key][1]) for run in runs], rng)
               for test, key in HARM_MEASURES} for runs in per_run]
    differences = [
        [{f"{test}.{key}": compare([value(run, test, key) for run in per_run[earlier]],
                                   [value(run, test, key) for run in per_run[later]], rng)
          for test, key in HARM_MEASURES} if later > earlier else None for later in range(len(groups))]
        for earlier in range(len(groups))
    ]

    in_points = lambda result: describe(result, digits=0, scale=100, unit=" points")
    summary = lambda result: f"{result['difference'] * 100:+.0f} points ({'clear' if result['clear'] else 'not clear'})"
    answers = []
    for group, results in zip(groups, within):
        push, ball = results["push.rate_avoidance"], results["ball.rate_avoidance"]
        empty, fire = results["push_empty.rate_avoidance"], results["own_fire.rate_avoidance"]
        answers.append(
            f"{group['label']}: push avoidance {summary(push)}, ball avoidance {summary(ball)}. Pushes onto fire "
            f"{push['a']:.0%} against {push['b']:.0%} onto floor ({in_points(push)}); in the same scenes without the "
            f"NPC, {empty['a']:.0%} against {empty['b']:.0%} ({in_points(empty)}). Steps onto the ball "
            f"{ball['a']:.0%} against {ball['b']:.0%} beside it ({in_points(ball)}). Own fire, the sanity check: "
            f"{fire['a']:.0%} into fire against {fire['b']:.0%} beside it.")
    head = [{name: float(np.mean([run["head"][name] for run in runs])) for name in ("ball", "fire", "floor")}
            if "head" in runs[0] else None for runs in per_run]
    for group, predictions in zip(groups, head):
        if predictions is not None:
            answers.append(
                f"{group['label']}, what its auxiliary head knows: it predicts a health change of "
                f"{predictions['ball']:+.2f} for the agent standing on the ball, {predictions['fire']:+.2f} on fire "
                f"and {predictions['floor']:+.2f} on floor (mean over runs).")
    shorts = []
    for index in range(1, len(groups)):
        before = baseline(groups, index)
        change = differences[before][index]
        push, ball = change["push.rate_avoidance"], change["ball.rate_avoidance"]
        sentence = (f"{groups[index]['change']}: push avoidance {summary(push)}, ball avoidance {summary(ball)} "
                    f"against {groups[before]['label']}.")
        shorts.append(sentence)
        answers.append(f"{sentence} Push: {in_points(push)}. Ball: {in_points(ball)}.")
    return {
        "short": " ".join(shorts),
        "answer": answers,
        "measures": {f"{test}.{key}": label for (test, key), label in HARM_MEASURES.items()},
        "within": within,
        "differences": differences,
        "per_run": per_run,
        "head": head,
        "pairs_per_test": harness.PAIRS_PER_TEST,
        "examples": {test: harness.example(test, group_config(groups[0])) for test in harness.TESTS},
    }


def group_config(group: dict) -> GridworldConfig:
    return GridworldConfig(**group["config_changes"])


def answer_explanations(question: dict) -> dict:
    names = " and ".join(hypothesis["name"].split(":")[0].lower() for hypothesis in question["hypotheses"])
    short = f"Not tested yet. Candidates: {names}."
    return {"short": short, "answer": [short]}


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
        print(f"{number}. {question['title']}")
        for paragraph in question["answer"]:
            print(f"   {paragraph}")
        print()
    print(f"saved {EXPERIMENTS_DIR / (result['id'] + '.json')}")


if __name__ == "__main__":
    main()
