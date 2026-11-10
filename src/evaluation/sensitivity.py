"""Sensitivity analyses of the scored runs: judge vote rules, vote ties, truncation of the
attached sources, matched-unit transitions from the specialist answers to the final answer,
and task-level rubric transitions.

    poetry run python -m src.evaluation.sensitivity output/evaluation \
        eval/reference_units/releases/triadcare/test
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.agent.results import RECORD_LIMIT, SOURCES_LIMIT
from src.evaluation.compare import RESAMPLES, SEED
from src.evaluation.judge import STATEMENT_DEFECTS
from src.evaluation.score import ESSENTIAL_STAGES

CONDITIONS = ("full", "full_ungoverned", "single_agent")
DEFECTS = {*STATEMENT_DEFECTS, "unavailable_citation"}
RULES = ("majority", "unanimous", "any")
STATUSES = ("retained", "omitted", "incorrect")


def essential_units(release: Path) -> set[tuple[str, str, str]]:
    """(task, stage, unit) of each required essential unit. A row can override the priority."""
    keys = set()
    for path in release.glob("*.reference.json"):
        reference = json.loads(path.read_text())
        units = {unit["id"]: unit for unit in reference["units"]}
        for row in reference["requirements"]:
            priority = row.get("priority", units[row["unit_id"]].get("priority"))
            if row["required"] and priority == "essential":
                keys.add((row["task_id"], row["stage"], row["unit_id"]))
    return keys


def runs(root: Path, condition: str):
    for path in sorted(root.glob(f"{condition}-r*/*/*/scores.json")):
        yield path, json.loads(path.read_text())


def comparisons(scores: dict, stage: str) -> dict[str, dict]:
    block = scores["preservation"].get(stage, {})
    return {c["task_id"]: c for c in block.get("comparisons", [])}


def retained(unit: dict, rule: str) -> bool:
    votes = sum(v == "retained" for v in unit["votes"])
    return {"majority": unit["status"] == "retained", "unanimous": votes == 3,
            "any": votes >= 1}[rule]


def ungrounded(statement: dict, rule: str) -> bool:
    votes = sum(v in DEFECTS for v in statement["votes"])
    return {"majority": statement["ungrounded"] is True, "unanimous": votes == 3,
            "any": votes >= 1}[rule]


def vote_rule_counts(root: Path, essential: set) -> tuple[dict, Counter]:
    """condition -> (scenario, task) -> measure -> [count, total], and the statement ties."""
    rows = {c: defaultdict(lambda: defaultdict(lambda: [0, 0])) for c in CONDITIONS}
    ties: Counter = Counter()
    for condition in CONDITIONS:
        for _, scores in runs(root, condition):
            scenario = scores["scenario_id"]
            for entry in scores["essential"]:
                if entry["status"] != "scored":
                    continue
                task = entry["task_id"]
                stage = ESSENTIAL_STAGES.get(entry["kind"], "source_to_final")
                units = comparisons(scores, stage)[task]["units"]
                ids = [u for u in units if (task, stage, u) in essential]
                for rule in RULES:
                    cell = rows[condition][(scenario, task)][f"Essential units retained ({rule})"]
                    cell[0] += sum(retained(units[u], rule) for u in ids)
                    cell[1] += len(ids)
            for entry in scores["quality"]:
                grounding = entry.get("grounding") or {}
                if grounding.get("status") != "scored":
                    continue
                statements = grounding["statements"]
                ties.update(len(set(s["votes"])) for s in statements)
                for rule in RULES:
                    cell = rows[condition][(scenario, entry["task_id"])][
                        f"Grounded answers (defect: {rule})"]
                    cell[0] += not any(ungrounded(s, rule) for s in statements)
                    cell[1] += 1
    return rows, ties


def pooled(rows: dict, keys: list, name: str) -> float:
    count = sum(rows[k][name][0] for k in keys)
    total = sum(rows[k][name][1] for k in keys)
    return count / total


def contrast(rows: dict, first: str, second: str, name: str) -> str:
    """Difference with the scenario bootstrap of compare.py (same seed and sample order)."""
    a, b = rows[first], rows[second]
    keys = sorted(set(a) & set(b))
    scenarios = sorted({s for s, _ in keys})
    by_scenario = {s: [k for k in keys if k[0] == s] for s in scenarios}
    rng = random.Random(SEED)
    samples = [[rng.choice(scenarios) for _ in scenarios] for _ in range(RESAMPLES)]
    diffs = sorted(
        pooled(a, picked, name) - pooled(b, picked, name)
        for picked in ([k for s in sample for k in by_scenario[s]] for sample in samples)
    )
    low, high = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    x, y = pooled(a, keys, name), pooled(b, keys, name)
    return f"{x:.1%} vs {y:.1%}: {x - y:+.1%} [{low:.1%}, {high:.1%}]"


def truncation(root: Path) -> str:
    """How often the attached sources of the full condition reached their limits."""
    consultations = records = cut = full = 0
    for path in root.glob("full-r*/*/*/tasks/*.json"):
        for step in json.loads(path.read_text()).get("trace") or []:
            result = step.get("observation")
            if not (str(step.get("tool", "")).startswith("consult_") and isinstance(result, dict)
                    and result.get("cited_sources")):
                continue
            text = result["cited_sources"]
            consultations += 1
            records += sum(line.startswith(("[", "- [")) for line in text.splitlines())
            cut += text.count(" [cut]")
            full += "[more sources cut]" in text
    return (f"{consultations} consultations, {records} attached records, {cut} cut at "
            f"{RECORD_LIMIT:,} characters, {full} consultations cut at {SOURCES_LIMIT:,}")


def unit_transitions(root: Path, condition: str) -> dict[str, dict[str, int]]:
    """Status of each unit in the specialist answers -> its status in the final answer,
    matched on run, task, and unit."""
    pairs: Counter = Counter()
    for _, scores in runs(root, condition):
        final = comparisons(scores, "source_to_final")
        for stage in ("ledger_to_context", "source_to_specialist"):
            for task, comparison in comparisons(scores, stage).items():
                answer = final.get(task)
                if not answer or "scored" != answer.get("status") or "scored" != comparison.get("status"):
                    continue
                for unit, verdict in comparison["units"].items():
                    if unit in answer["units"]:
                        pairs[verdict["status"], answer["units"][unit]["status"]] += 1
    return {a: {b: pairs[a, b] for b in STATUSES} for a in STATUSES}


def rubric_transitions(root: Path) -> dict[str, int]:
    """single_agent -> full rubric decision per task, with the most frequent decision of the runs."""
    decisions = {c: defaultdict(list) for c in ("full", "single_agent")}
    for condition, by_task in decisions.items():
        for _, scores in runs(root, condition):
            for entry in scores["quality"]:
                if entry.get("status") == "scored":
                    by_task[scores["scenario_id"], entry["task_id"]].append(entry["decision"])
    pairs: Counter = Counter()
    for task, full in decisions["full"].items():
        single = decisions["single_agent"][task]
        pairs[Counter(single).most_common(1)[0][0], Counter(full).most_common(1)[0][0]] += 1
    return {f"{a} -> {b}": n for (a, b), n in sorted(pairs.items())}


def report(root: Path, release: Path) -> str:
    rows, ties = vote_rule_counts(root, essential_units(release))
    lines = [f"# Sensitivity analyses ({root.name})", "", "## Judge vote rules", ""]
    for name in sorted({n for task in rows["full"].values() for n in task}):
        lines.append(f"- {name}")
        for other in ("single_agent", "full_ungoverned"):
            lines.append(f"  - full - {other}: {contrast(rows, 'full', other, name)}")
    lines += ["", f"Statement votes by number of distinct verdicts: {dict(sorted(ties.items()))}",
              "", "## Attached sources", "", truncation(root), "",
              "## Specialist answer -> final answer (unit status)", ""]
    for condition in ("full", "full_ungoverned"):
        lines.append(f"- {condition}: {unit_transitions(root, condition)}")
    lines += ["", "## Rubric decision per task, single_agent -> full", "",
              json.dumps(rubric_transitions(root), indent=2)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("release", type=Path, help="Release split directory with *.reference.json")
    args = parser.parse_args(argv)
    text = report(args.root, args.release)
    (args.root / "sensitivity.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
