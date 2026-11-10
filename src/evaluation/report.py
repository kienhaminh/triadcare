"""Sum scored runs into the manuscript tables, by model and by scenario.

    poetry run python -m src.evaluation.report output/evaluation/dev-smoke

Unexecuted, unscored, and runtime-unsupported items stay in the denominator.
Unavailable imaging reuse opportunities leave the denominator and are counted separately.
An RQ3 stage not performed after an earlier stage stopped leaves the denominator,
except for the full workflow, which includes all assigned sequences.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.release import PRESERVATION_STAGES
from src.evaluation.run import dump
from src.evaluation.score import ARTIFACT_NOTES, STAGE_ARTIFACTS

STAGE_ROWS = {
    "conversation_to_derived": "Conversations to recaps",
    "ledger_to_context": "PKL to DoctorAgent context",
    "source_to_specialist": "Sources to specialist-agent results",
    "sources_to_proposal": "Sources to proposal",
    "proposal_to_revision": "Proposal to revision",
    "source_to_final": "Sources to final answer",
}
LABEL_ROWS = {
    "source_link": "Units with missing or incorrect source links",
    "attribution": "Units with incorrect reporter attribution",
    "temporal": "Units with temporal errors",
    "uncertainty": "Units with uncertainty errors",
}
# Manuscript names and column order of the model configurations.
MODEL_NAMES = {
    "openai/gpt-6-luna": "GPT-6 Luna",
}
DIMENSION_ROWS = {
    "source_support": "Source support",
    "coverage": "Required coverage",
    "attribution": "Attribution",
    "temporal_fidelity": "Temporal fidelity",
    "uncertainty": "Uncertainty",
    "task_completion": "Task fulfillment",
}
RQ3_ROWS = {
    "source_open": "Correct selected source opened (review API proxy)",
    "rejection": "Rejection and reason recorded",
    "revision_link": "Revision linked to rejected proposal",
    "content_gate": "Code-check approval rule followed",
    "approval_binding": "Approval bound to correct version and content",
    "fhir_write": "Successful FHIR write after approval",
    "full_workflow": "Complete update sequence (all assigned)",
    "caregiver_denial": "Caregiver write request blocked",
    "audit": "Audit complete for performed stages",
}


def _ratio(count: int, total: int, **other: int) -> dict[str, Any]:
    return {"count": count, "total": total, **{key: value for key, value in other.items() if value}}


def summarize(runs: list[dict]) -> dict[str, Any]:
    tasks = Counter(status for run in runs for status in run["tasks"].values())
    quality = [entry for run in runs for entry in run["quality"]]
    scored = [entry for entry in quality if entry.get("status") == "scored"]
    decisions = Counter(entry["decision"] for entry in scored)
    needs = Counter(entry["status"] for run in runs for entry in run["source_needs"])
    specialist = Counter(entry["status"] for run in runs for entry in run["specialist_results"])
    call_opportunities = [entry for run in runs for entry in run["specialist_calls"]]
    calls = Counter(
        call["status"] for entry in call_opportunities for call in entry.get("calls", [])
    )
    imaging = {
        metric: Counter(
            entry["status"] for run in runs for entry in run["imaging"] if entry["metric"] == metric
        )
        for metric in ("imaging_reuse", "imaging_rerun")
    }
    storage = [entry for run in runs for entry in run["storage"]]

    coordination = {
        "Tasks with outputs": _ratio(
            tasks["completed"],
            sum(tasks.values()),
            execution_error=tasks["execution_error"],
            unreached=tasks["unreached"],
            not_run=tasks["not_run"],
        ),
        "Scored tasks": _ratio(
            len(scored),
            len(quality),
            unscored=sum(entry.get("status") == "unscored" for entry in quality),
        ),
        "Required source needs fulfilled": _ratio(
            needs["fulfilled"], sum(needs.values()), not_executed=needs["not_executed"]
        ),
        "Correctly incorporated specialist-agent results": _ratio(
            specialist["incorporated"],
            sum(specialist.values()),
            unscored=specialist["unscored"],
            not_executed=specialist["not_executed"],
        ),
        "Specialist-agent calls outside the required needs (code)": {
            "count": calls["unnecessary"],
            "specialist_calls": sum(calls.values()),
            "unscored_calls": calls["unscored"],
            "exercised_opportunities": sum(e["status"] == "exercised" for e in call_opportunities),
            "opportunities": len(call_opportunities),
        },
        "Correct imaging-result reuse": {
            **_ratio(
                imaging["imaging_reuse"]["pass"],
                sum(imaging["imaging_reuse"].values()) - imaging["imaging_reuse"]["not_available"],
                unscored=imaging["imaging_reuse"]["unscored"],
                not_executed=imaging["imaging_reuse"]["not_executed"],
            ),
            "not_available": imaging["imaging_reuse"]["not_available"],
        },
        "Reanalyses performed": _ratio(
            imaging["imaging_rerun"]["pass"],
            sum(imaging["imaging_rerun"].values()),
            not_executed=imaging["imaging_rerun"]["not_executed"],
        ),
    }

    preservation: dict[str, Any] = {
        "Stored conversations intact": _ratio(sum(e["passed"] for e in storage), len(storage))
    }
    label_counts: Counter = Counter()
    final_total = final_unscored = 0
    for stage in PRESERVATION_STAGES:
        counts: Counter = Counter()
        artifact = None
        for run in runs:
            block = run["preservation"][stage]
            artifact = block["artifact"]
            for comparison in block["comparisons"]:
                if comparison["status"] == "scored":
                    for unit in comparison["units"].values():
                        counts[unit["status"]] += 1
                        if stage == "source_to_final":
                            label_counts.update(k for k, v in unit["labels"].items() if v)
                            if unit["status"] == "retained" and unit["labels"].get("detail"):
                                counts["retained_detail"] += 1
                else:
                    counts[comparison["status"]] += comparison["required"]
                counts["required"] += comparison["required"]
        if stage == "source_to_final":
            final_total, final_unscored = counts["required"], counts["unscored"]
        preservation[STAGE_ROWS[stage]] = {
            "artifact": artifact,
            "required": counts["required"],
            "retained": counts["retained"],
            "omitted": counts["omitted"],
            "incorrect": counts["incorrect"],
            **{
                key: counts[key]
                for key in ("retained_detail", "not_executed", "unscored", "runtime_unsupported")
                if counts[key]
            },
        }

    output_quality: dict[str, Any] = essential_rows([e for run in runs for e in run["essential"]])
    output_quality.update(
        {
            row: _ratio(label_counts[label], final_total, unscored_units=final_unscored)
            for label, row in LABEL_ROWS.items()
        }
    )
    output_quality["Unsupported claims"] = {
        "count": sum(entry["claims_unsupported"] for entry in scored),
        "total": sum(entry["claims_assessed"] for entry in scored),
    }
    not_scored = {
        "unscored": sum(entry.get("status") == "unscored" for entry in quality),
        "not_executed": sum(entry.get("status") == "not_executed" for entry in quality),
        "no_stored_proposal": sum(entry.get("status") == "no_output" for entry in quality),
    }
    # Grounding: an answer is grounded when the statement judge finds no ungrounded statement.
    grounding = [entry.get("grounding") or {} for entry in quality]
    judged = [g for g in grounding if g.get("status") == "scored"]
    ungrounded = [sum(s["ungrounded"] is True for s in g["statements"]) for g in judged]
    verified = [s for g in judged for s in g["statements"] if s["ungrounded"] is not None]
    output_quality["Grounded answers"] = _ratio(
        sum(n == 0 for n in ungrounded), len(quality),
        **{**not_scored, "unscored": sum(g.get("status") == "unscored" for g in grounding)},
    )
    output_quality["Grounded answers"]["rate"] = 1
    output_quality["Ungrounded statements"] = {"count": sum(ungrounded), "total": len(verified)}
    # Uncited claims cannot be checked against the cited sources; they are counted apart.
    output_quality["Uncited claims"] = {
        "count": sum(s["ungrounded"] is None for g in judged for s in g["statements"]),
        "total": sum(len(g["statements"]) for g in judged),
    }
    for decision, row in (
        ("full_pass", "Full passes"),
        ("partial_pass", "Partial passes"),
        ("failure", "Failures"),
    ):
        output_quality[row] = _ratio(decisions[decision], len(quality), **not_scored)
    output_quality["Full passes"]["rate"] = 1
    output_quality["Outputs with critical errors"] = _ratio(
        sum(bool(entry["critical_errors"]) for entry in scored), len(quality), **not_scored
    )
    # Score distribution per rubric dimension, over scored outputs: a full pass needs all six.
    for key, row in DIMENSION_ROWS.items():
        given = Counter(entry["scores"].get(key) for entry in scored)
        output_quality[f"{row} (2 / 1 / 0)"] = {"scores": [given[2], given[1], given[0]]}

    interaction = interaction_rows([entry for run in runs for entry in run["rq3"]])
    interaction.update(revision_unit_rows([run.get("revision_units") for run in runs]))
    checks = [entry for run in runs for entry in run.get("citations", [])]
    citations = {
        "DoctorAgent cited IDs that no record tool returned": _ratio(
            sum(len(e["answer"]["unresolved"]) for e in checks),
            sum(e["answer"]["cited"] for e in checks),
        ),
    }
    for name in sorted({n for e in checks for n in e["specialists"]}):
        rows = [e["specialists"][name] for e in checks if name in e["specialists"]]
        citations[f"{name} cited IDs that its record tools did not return"] = _ratio(
            sum(len(r["unresolved"]) for r in rows), sum(r["cited"] for r in rows)
        )
    return {
        "provenance": provenance_rows([e for run in runs for e in run.get("provenance", [])]),
        "coordination": coordination,
        "preservation": preservation,
        "output_quality": output_quality,
        "interaction": interaction,
        "citations": citations,
    }


def essential_rows(entries: list[dict]) -> dict[str, Any]:
    """Essential units retained in final answers, in total and by task kind.

    Unscored or unexecuted tasks keep their units in the total, as not retained.
    """

    def row(selected: list[dict]) -> dict[str, Any]:
        return {
            **_ratio(
                sum(e["retained"] or 0 for e in selected),
                sum(e["essential"] for e in selected),
                not_scored_units=sum(e["essential"] for e in selected if e["retained"] is None),
            ),
            "rate": 1,
        }

    rows = {"Essential units retained in final answers": row(entries)}
    for kind in dict.fromkeys(e["kind"] for e in entries):
        rows[f"  {kind}"] = row([e for e in entries if e["kind"] == kind])
    return rows


def revision_unit_rows(entries: list[dict | None]) -> dict[str, Any]:
    """Revision content split into W2-only reference units and kept earlier content."""
    scored = [e for e in entries if e and e["status"] == "scored"]
    if not scored:
        return {}
    return {
        "W2-only reference units retained": _ratio(
            sum(e["changes"]["count"] for e in scored), sum(e["changes"]["total"] for e in scored)
        ),
        "Earlier supported content kept in revision (units)": _ratio(
            sum(e["kept"]["count"] for e in scored), sum(e["kept"]["total"] for e in scored)
        ),
    }


def interaction_rows(entries: list[dict]) -> dict[str, Any]:
    """RQ3 rows. The full workflow includes every assigned sequence."""
    interaction: dict[str, Any] = {}
    for metric, row in RQ3_ROWS.items():
        states = Counter(entry["status"] for entry in entries if entry["metric"] == metric)
        if not states:
            continue
        if metric == "full_workflow":
            selected = [entry for entry in entries if entry["metric"] == metric]
            interaction[row] = {
                "count": states["pass"], "total": len(selected),
                "completed": states["pass"],
                "exercised": sum(entry.get("exercised", entry["status"] not in {"not_performed", "runtime_unsupported"})
                                 for entry in selected),
                "not_performed": states["not_performed"], "unscored": states["unscored"],
                "runtime_unsupported": states["runtime_unsupported"],
                "reasons": dict(Counter(entry.get("reason", "unspecified") for entry in selected
                                        if entry["status"] in {"not_performed", "unscored", "runtime_unsupported"})),
            }
            continue
        skipped = states.pop("not_performed", 0)
        if not states:
            interaction[row] = {"status": "not_performed", "not_performed": skipped}
        elif states.get("runtime_unsupported") == sum(states.values()):
            interaction[row] = {"status": "runtime_unsupported", "total": sum(states.values())}
        else:
            interaction[row] = _ratio(
                states["pass"],
                sum(states.values()),
                not_performed=skipped,
                **{key: value for key, value in states.items() if key not in {"pass", "fail"}},
            )
    return interaction


def provenance_rows(entries: list[dict]) -> dict[str, Any]:
    """Stage counts of the essential source anchors, per source kind. Anchors of tasks
    that did not run stay in the totals."""
    essential = [e for e in entries if e.get("priority", "essential") == "essential"]
    skipped = [e for e in essential if e["status"] == "not_executed"]
    rows: dict[str, Any] = {}
    for kind in ("check-in", "fhir", "imaging"):
        mine = [e for e in essential if e.get("kind") == kind]
        total = len(mine)
        if not mine:
            continue
        returned = [e for e in mine if e["returned"]]
        final = [e for e in mine if e["final_cited"]]
        rows[f"{kind}: returned by a record tool"] = _ratio(len(returned), total)
        rows[f"{kind}: cited in the output"] = _ratio(len(final), total)
        rows[f"{kind}: returned, not cited in the output"] = _ratio(
            sum(not e["final_cited"] for e in returned), len(returned)
        )
        if any(e["specialist_cited"] is not None for e in mine):
            specialist = [e for e in mine if e["specialist_cited"]]
            rows[f"{kind}: returned, not cited by a specialist"] = _ratio(
                sum(not e["specialist_cited"] for e in returned), len(returned)
            )
            rows[f"{kind}: cited by a specialist, not cited in the output"] = _ratio(
                sum(not e["final_cited"] for e in specialist), len(specialist)
            )
        if kind == "check-in":
            rows[f"{kind}: cited in the output, exact or same session"] = _ratio(
                sum(e["final_cited"] or e["final_near_miss"] for e in mine), total
            )
    for row in rows.values():
        row["rate"] = 1
    if skipped:
        rows["Essential anchors in tasks not executed"] = {"count": len(skipped)}
    return rows


def build_report(root: Path) -> dict[str, Any]:
    runs = [json.loads(path.read_text()) for path in sorted(root.rglob("scores.json"))]
    # The manuscript reports runs of different system or scoring versions separately.
    for key, label in (
        ("judge", "judges"), ("system", "system versions"),
        ("release_manifest_sha256", "reference releases"),
        ("quality_source_scope", "quality evidence scopes"),
    ):
        versions = {json.dumps(run.get(key), sort_keys=True) for run in runs}
        if len(versions) > 1:
            raise ValueError(f"Runs under {root} use different {label}; report them separately.")
    order = list(MODEL_NAMES)
    models = sorted(
        {run["model"] for run in runs},
        key=lambda m: (order.index(m) if m in order else len(order), m),
    )
    report: dict[str, Any] = {
        "models": {},
        "judge": runs[0].get("judge") if runs else None,
        "quality_source_scope": runs[0].get("quality_source_scope") if runs else None,
        "system": runs[0].get("system") if runs else None,
        "stage_artifacts": {
            STAGE_ROWS[stage]: ARTIFACT_NOTES.get(artifact, "Runtime has no such artifact.")
            for stage, artifact in STAGE_ARTIFACTS.items()
        },
    }
    for model in models:
        mine = [run for run in runs if run["model"] == model]
        report["models"][model] = {
            "splits": sorted({run["split"] for run in mine}),
            "scenarios": sorted(run["scenario_id"] for run in mine),
            "all": summarize(mine),
            "by_scenario": {run["scenario_id"]: summarize([run]) for run in mine},
        }
    return report


def _cell(value: dict[str, Any]) -> str:
    if value.get("status") == "runtime_unsupported":
        return f"runtime unsupported (0/{value['total']} measurable)"
    if value.get("status") == "not_performed":
        return "\u2014"
    if "scores" in value:
        return " / ".join(map(str, value["scores"]))
    if "retained" in value:
        if not value["required"]:
            return "no opportunity"
        text = (
            f"{value['retained']} / {value['omitted']} / {value['incorrect']} of "
            f"{value['required']} ({value['retained'] / value['required']:.1%})"
        )
    elif "specialist_calls" in value:
        text = f"{value['count']} of {value['specialist_calls']} calls"
        value = {k: v for k, v in value.items() if k != "specialist_calls"}
    elif "total" in value:
        if not value["total"]:
            if "not_available" not in value:
                return "no opportunity"
            text = "no opportunity"
        else:
            text = f"{value['count']}/{value['total']}"
        if value.get("rate") and value["total"]:
            text += f" ({value['count'] / value['total']:.1%})"
    else:
        text = str(value["count"])
    extras = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "count",
            "total",
            "rate",
            "retained",
            "omitted",
            "incorrect",
            "required",
            "artifact",
        }
    }
    if extras:
        text += (
            " ("
            + ", ".join(f"{key.replace('_', ' ')} {item}" for key, item in extras.items())
            + ")"
        )
    return text


def markdown(report: dict[str, Any]) -> str:
    models = list(report["models"])
    lines = ["# Manuscript measures", ""]
    lines += [
        (
            "Scenarios are synthetic and not clinically validated. The measures cover "
            "information completeness and correctness against the sources, not clinical safety."
        ),
        "Tasks within a scenario are correlated. Use scenario-level groups for inference.", "",
    ]
    if report["judge"]:
        lines += [f"Judge: `{json.dumps(report['judge'])}`", ""]
    if report.get("quality_source_scope") == "cited_sources_v1":
        lines += ["Quality judgments check cited sources only, not the complete record.", ""]
    if report["system"]:
        lines += [f"System: `{json.dumps(report['system'])}`", ""]
    for model in models:
        entry = report["models"][model]
        lines.append(f"- `{model}`: {entry['splits']} {', '.join(entry['scenarios'])}")
    titles = {
        "provenance": "Source provenance through the stages (code, essential anchors)",
        "coordination": "RQ1: coordination and evidence use",
        "preservation": "RQ2: information preservation (retained / omitted / incorrect)",
        "output_quality": "RQ2: error labels and output quality",
        "interaction": "RQ3: inspection, revision, and authorized actions",
        "citations": "Source-citation checks (code)",
    }
    for section, title in titles.items():
        lines += ["", f"## {title}", ""]
        rows = report["models"][models[0]]["all"][section] if models else {}
        if not rows:
            lines.append("No opportunity in the scored scenarios.")
            continue
        names = [MODEL_NAMES.get(m, m) for m in models]
        lines += ["| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(models)]
        for row in rows:
            cells = [_cell(report["models"][m]["all"][section][row]) for m in models]
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
        if section == "preservation":
            lines += ["", "Artifact compared at each stage:", ""]
            lines += [f"- {row}: {note}" for row, note in report["stage_artifacts"].items()]
    headline = (
        ("coordination", "Required source needs fulfilled"),
        ("coordination", "Correctly incorporated specialist-agent results"),
        ("preservation", STAGE_ROWS["source_to_final"]),
        ("output_quality", "Grounded answers"),
        ("output_quality", "Essential units retained in final answers"),
        ("output_quality", "Full passes"),
    )
    lines += ["", "## By scenario", ""]
    lines += ["| Model | Scenario | " + " | ".join(row for _, row in headline) + " |"]
    lines += ["|---|---|" + "---|" * len(headline)]
    for model in models:
        for scenario, summary in report["models"][model]["by_scenario"].items():
            cells = [_cell(summary[section][row]) for section, row in headline]
            lines.append(
                f"| {MODEL_NAMES.get(model, model)} | {scenario} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        (
            "Rates divide by the totals shown. Unavailable imaging reuse opportunities are "
            "excluded from the total and listed in brackets. Items that were not executed, not scored, "
            "or not supported by the runtime stay in the totals and are listed in brackets. "
            "Retained detail: retained units that leave out a secondary qualifier; the strict "
            "retained count subtracts them. Rubric rows count scored outputs per score."
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path, help="scored output directory")
    args = parser.parse_args(argv)
    report = build_report(args.out)
    dump(args.out / "tables.json", report)
    (args.out / "tables.md").write_text(markdown(report))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
