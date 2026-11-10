"""Paired comparison of two conditions over repeated runs.

Each measure is a pooled ratio (count over total) or a mean per task. The comparison
pairs the two conditions on the same tasks. Runs of one condition are pooled. The
confidence interval resamples scenarios (the clusters), with all their tasks and runs.

    poetry run python -m src.evaluation.compare output/evaluation/v13-test full single_agent
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

RESAMPLES = 10000
SEED = 7


def task_counts(scores: dict, run_dir: Path) -> dict[str, dict[str, float]]:
    """Counts of each measure per task of one scenario run."""
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(task_id: str, name: str, count: float, total: float = 1) -> None:
        counts[task_id][name] += count
        counts[task_id][name + "#total"] += total

    for entry in scores["quality"]:
        grounding = entry.get("grounding") or {}
        if grounding.get("status") == "scored":
            statements = grounding["statements"]
            bad = sum(s["ungrounded"] is True for s in statements)
            verified = sum(s["ungrounded"] is not None for s in statements)
            add(entry["task_id"], "Grounded answers", bad == 0)
            add(entry["task_id"], "Ungrounded lines", bad, verified)
            add(entry["task_id"], "Uncited claims", len(statements) - verified, len(statements))
        if entry.get("status") == "scored":
            add(entry["task_id"], "Full passes", entry["decision"] == "full_pass")
    for entry in scores["essential"]:
        if entry.get("status") == "scored":
            add(entry["task_id"], "Essential units retained", entry["retained"], entry["essential"])
    for entry in scores["source_needs"]:
        add(entry["task_id"], "Source needs fulfilled", entry["status"] == "fulfilled")
    for entry in scores.get("provenance", []):
        if entry["status"] != "scored" or entry["priority"] != "essential":
            continue
        kind = entry["kind"]
        add(entry["task_id"], f"{kind}: returned", entry["returned"])
        add(entry["task_id"], f"{kind}: cited in the output", entry["final_cited"])
    for path in (run_dir / "tasks").glob("*.json"):
        record = json.loads(path.read_text())
        # Count only the tasks of the scored release; a dropped task has no score.
        if record.get("status") != "completed" or record.get("task_id") not in scores["tasks"]:
            continue
        usage = record.get("usage") or {}
        add(record["task_id"], "Tokens per task", usage.get("total_tokens", 0))
        add(record["task_id"], "Requests per task", record.get("requests") or 0)
        add(record["task_id"], "Seconds per task", record.get("duration_seconds") or 0)
    return counts


def load(root: Path, condition: str) -> dict[tuple[str, str], list[dict]]:
    """(scenario, task) -> the task counts of each run of one condition."""
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(root.glob(f"{condition}-r*/*/*/scores.json")):
        scores = json.loads(path.read_text())
        for task_id, counts in task_counts(scores, path.parent).items():
            rows[(scores["scenario_id"], task_id)].append(counts)
    return rows


def pooled(rows: dict, keys: list, name: str) -> float | None:
    count = sum(c.get(name, 0) for key in keys for c in rows.get(key, []))
    total = sum(c.get(name + "#total", 0) for key in keys for c in rows.get(key, []))
    return count / total if total else None


def compare(root: Path, first: str, second: str) -> str:
    paths = [path for condition in (first, second)
             for path in root.glob(f"{condition}-r*/*/*/scores.json")]
    scopes = {json.loads(path.read_text()).get("quality_source_scope") for path in paths}
    if len(scopes) > 1:
        raise ValueError("Runs use different quality evidence scopes; compare them separately.")
    a, b = load(root, first), load(root, second)
    keys = sorted(set(a) & set(b))
    scenarios = sorted({s for s, _ in keys})
    by_scenario = {s: [k for k in keys if k[0] == s] for s in scenarios}
    names = list(dict.fromkeys(n for rows in (a, b) for runs in rows.values()
                               for c in runs for n in c if not n.endswith("#total")))
    runs = {c: len({p.parent.parent.parent.name for p in root.glob(f"{c}-r*/*/*/scores.json")})
            for c in (first, second)}
    rng = random.Random(SEED)
    samples = [[rng.choice(scenarios) for _ in scenarios] for _ in range(RESAMPLES)]
    lines = [
        f"# {first} vs {second} ({root.name})", "",
        (
            f"Runs: {first} {runs[first]}, {second} {runs[second]}. Paired tasks: {len(keys)} "
            f"in {len(scenarios)} scenarios. Ratios pool all runs. The 95% interval resamples "
            f"scenarios ({RESAMPLES} resamples). Per scenario: the sign of {first} minus {second}."
        ),
        "", f"| Measure | {first} | {second} | Difference | 95% interval | Per scenario |",
        "|---|---|---|---|---|---|",
    ]
    if scopes == {"cited_sources_v1"}:
        lines[2:2] = ["Quality judgments check cited sources only, not the complete record.", ""]
    for name in names:
        x, y = pooled(a, keys, name), pooled(b, keys, name)
        if x is None or y is None:
            continue
        diffs = []
        for sample in samples:
            picked = [k for s in sample for k in by_scenario[s]]
            dx, dy = pooled(a, picked, name), pooled(b, picked, name)
            if dx is not None and dy is not None:
                diffs.append(dx - dy)
        diffs.sort()
        low, high = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
        signs = ""
        for s in scenarios:
            dx, dy = pooled(a, by_scenario[s], name), pooled(b, by_scenario[s], name)
            signs += "." if dx is None or dy is None else "+" if dx > dy else "-" if dx < dy else "="
        fmt = (lambda v: f"{v:,.1f}") if name.endswith("per task") else (lambda v: f"{v:.1%}")
        lines.append(
            f"| {name} | {fmt(x)} | {fmt(y)} | {fmt(x - y)} | {fmt(low)} to {fmt(high)} | {signs} |"
        )
    lines += ["", "Scenario order for the signs: " + ", ".join(scenarios)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args(argv)
    text = compare(args.root, args.first, args.second)
    (args.root / f"compare-{args.first}-{args.second}.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
