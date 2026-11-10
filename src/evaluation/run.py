"""Run frozen scenario tasks with one model and save the artifacts for scoring.

Each (model, scenario) pair runs in its own process, so no process-global state
(environment, imaging cache) crosses runs. Each process imports its scenario
into a fresh ledger schema. The Doctor runs statelessly and synchronously
(no background tasks), so results do not depend on task wait timing.

    poetry run python -m src.evaluation.run --split development --scenarios quiet_week \\
        --models openai/gpt-6-luna --out output/evaluation/dev-smoke

The test split is the manuscript experiment. It runs only with --confirm-test.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.release import ROOT, Case, Release

# The runtime needs an explicit patient ID; the frozen prompts do not contain one.
# Every model receives the same framing.
REQUEST_TEMPLATE = "Patient ID: {patient_id}. Review date: {as_of}.\n\n{prompt}"
RUNTIME_MODE = (
    "stateless Doctor turn, synchronous consultations, no background tasks, "
    "no answer checks"
)

Respond = Callable[[list[dict]], dict]


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def request_text(case: Case, task: dict) -> str:
    return REQUEST_TEMPLATE.format(
        patient_id=case.patient_id, as_of=task["as_of"], prompt=task["prompt"]
    )


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


class Clinician(Protocol):
    """Scripted clinician actions around tasks, such as rejecting or authorizing a proposal."""

    def before(self, task: dict) -> dict | None: ...

    def after(self, task: dict, record: dict) -> dict | None: ...


def _hook(action: Callable[..., dict | None], *args: Any) -> dict | None:
    """A failed clinician action is an outcome of the task, not a failed run."""
    try:
        return action(*args)
    except Exception as error:  # noqa: BLE001
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def execute_tasks(
    case: Case,
    respond: Respond,
    clinician: Clinician | None = None,
    session_prefix: str | None = None,
) -> list[dict]:
    """Run tasks in frozen order. A task whose dependency did not complete is not run.

    With a session_prefix, the requests of each task session carry the OpenRouter
    session_id <prefix>:<session>, so a follow-up task shares its parent's session.
    """
    from src.agent.citations import citation_check
    from src.fhir.record_updates import REQUEST_ID
    from src.ledger.temporal import ledger_scope
    from src.llm.client import session as llm_session

    sessions: dict[str, list[dict]] = {}
    status: dict[str, str] = {}
    records = []
    for task in case.tasks:
        with ledger_scope(case.patient_id, task["as_of"]):
            dependency = task["depends_on"]
            record: dict[str, Any] = {
                "task_id": task["id"],
                "question_id": task["question_id"],
                "session_id": task["session_id"],
                "depends_on": dependency,
                "request": request_text(case, task),
            }
            if dependency and status.get(dependency) != "completed":
                record.update(status="unreached", reason=f"Dependency {dependency} did not complete.")
            else:
                if clinician is not None:
                    record["clinician_before"] = _hook(clinician.before, task)
                before = record.get("clinician_before") or {}
                if before.get("status") in {"error", "not_exercised", "revision_not_exercised", "unscored"}:
                    record.update(status="unreached", reason=before.get("reason") or before.get("error"))
                    status[task["id"]] = record["status"]
                    records.append(record)
                    continue
                if before.get("feedback") is not None:
                    record["revision_selection"] = before.get("selection")
                    record["effective_feedback"] = before["feedback"]
                    record["request"] = request_text(case, {**task, "prompt": before["feedback"]})
                history = sessions.get(task["session_id"], [])
                messages = [*history, {"role": "user", "content": record["request"]}]
                record["history_answers"] = [m["content"] for m in history if m["role"] == "assistant"]
                started = time.monotonic()
                request_token = REQUEST_ID.set(task["id"])
                result = None
                grouping = (
                    llm_session(f"{session_prefix}:{task['session_id']}")
                    if session_prefix
                    else nullcontext()
                )
                try:
                    with grouping:
                        result = respond(messages)
                    answer = (result.get("message") or {}).get("content") or ""
                    if result.get("finish_reason") != "stop" or not answer.strip():
                        raise RuntimeError(
                            f"No final answer (finish_reason={result.get('finish_reason')})."
                        )
                    record.update(
                        status="completed",
                        answer=answer,
                        trace=result.get("trace", []),
                        usage=result.get("usage"),
                        requests=result.get("requests"),
                        citation_check=citation_check(answer, result.get("trace", [])),
                    )
                    sessions[task["session_id"]] = [*messages, {"role": "assistant", "content": answer}]
                except Exception as error:  # noqa: BLE001 - every execution error is an outcome.
                    record.update(
                        status="execution_error",
                        error_type=type(error).__name__,
                        error=str(error),
                        trace=(result or {}).get("trace", getattr(error, "partial_trace", [])),
                        usage=(result or {}).get("usage", getattr(error, "partial_usage", None)),
                        requests=(result or {}).get(
                            "requests", getattr(error, "partial_requests", None)
                        ),
                    )
                    if result is not None:
                        record.update(
                            finish_reason=result.get("finish_reason"),
                            model=result.get("model"),
                        )
                        message = result.get("message")
                        answer = message.get("content") if isinstance(message, dict) else None
                        if isinstance(answer, str) and answer.strip():
                            record.update(partial_answer=answer, answer_complete=False)
                    elif hasattr(error, "partial_usage_complete"):
                        record["usage_complete"] = error.partial_usage_complete
                finally:
                    REQUEST_ID.reset(request_token)
                record["duration_seconds"] = round(time.monotonic() - started, 3)
                if clinician is not None:
                    record["clinician_after"] = _hook(clinician.after, task, record)
            status[task["id"]] = record["status"]
            records.append(record)
    return records


def fhir_root() -> str | None:
    """The FHIR root that requests of this run use; None when FHIR is not configured."""
    from src.config.fhir import fhir_settings

    config = fhir_settings()
    return config.root() if config.base_url is not None else None


def _git() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _drop_schema(schema: str) -> None:
    import psycopg
    from psycopg import sql

    from src.config.database import database_settings

    url = database_settings().url
    if not url:
        raise RuntimeError("Set DATABASE_URL to drop an evaluation schema.")
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
        )


def recap_scenario(store: Any, scenario_id: str) -> dict[str, Any]:
    """CareAgent recap of every imported check-in, in recording order.

    A failed recap is recorded and does not fail the run; Care Agent then reads that
    check-in directly.
    """
    from src.ledger.recap import RecapStore, recap_session

    recaps = RecapStore(store)
    sessions = sorted(
        {
            (row["source_day"], row["audience"], row["session_id"])
            for row in store.export_scenario(scenario_id)
        }
    )
    results = {}
    for _, _, session_id in sessions:
        try:
            results[session_id] = recap_session(recaps, session_id)["status"]
        except Exception as error:  # noqa: BLE001 - recorded per session.
            results[session_id] = f"error: {type(error).__name__}: {error}"
    return {
        "sessions": len(results),
        "stored": sum(status == "stored" for status in results.values()),
        "results": results,
    }


def run_case(
    case: Case,
    release: Release,
    model: str,
    out: Path,
    *,
    seed_fhir: bool,
    keep_ledger: bool,
    recap_only: bool = False,
    condition: str = "full",
) -> None:
    """Prepare isolated state, run all tasks of one scenario, and save artifacts.

    condition selects the experiment: "full" is the multi-agent system, "full_ungoverned"
    is the same system without source governance, and "single_agent" is the one-agent
    system."""
    from src.agent.doctor import respond
    from src.evaluation.clinician import ScriptedClinician, fixture_note, record_target
    from src.evaluation.judge_factory import judge_from_env
    from src.evaluation.judgments import Judgments
    from src.fhir.record_updates import (
        RecordStore,
        content_sha256,
        default_record_updates,
        note_text,
        reset_default_record_updates,
        resource_version,
    )
    from src.fhir.seed import seed_fhir as seed
    from src.ledger.store import LedgerStore
    from src.llm.client import request_profile
    from src.llm.client import session as llm_session

    schema = f"eval_{uuid.uuid4().hex[:20]}"
    # Proposals of this run, and a separate state for the independent caregiver-role probe.
    record_schema, probe_schema = f"{schema}_records", f"{schema}_probe"
    # This process runs only this case (--child), so the settings loaders read the
    # run's configuration from the environment.
    os.environ.update(
        LLM_MODEL=model,
        LEDGER_SCHEMA=schema,
        RECORD_SCHEMA=record_schema,
        IMAGING_CATALOG_PATH=str(case.imaging_catalog_path),
    )
    reset_default_record_updates()
    generation = request_profile()
    manifest: dict[str, Any] = {
        "split": case.split,
        "scenario_id": case.scenario_id,
        "patient_id": case.patient_id,
        "model": model,
        "reasoning_effort": generation["reasoning_effort"],
        # Generation parameters of every request.
        "generation": generation,
        "release_manifest_sha256": release.manifest_sha256,
        "git": _git(),
        "runtime_mode": RUNTIME_MODE,
        "condition": condition,
        "request_template": REQUEST_TEMPLATE,
        "ledger_schema": schema,
        "record_schemas": [record_schema, probe_schema],
        "fhir_root": fhir_root(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    target = record_target(case)
    # The scripted clinician gates approval on the judge. A later rescoring cannot replay
    # a write that an unscored gate withheld, so a record-target run needs the judge now.
    judge = judge_from_env(require_key=target is not None)
    dump(out / "run.json", {**manifest, "status": "running"})
    try:
        store = LedgerStore()
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(case.conversation_path, directory)
            manifest["ledger_import"] = store.import_corpus(Path(directory))
        dump(out / "ledger-readback.json", store.export_scenario(case.scenario_id))
        from src.ledger.recap import RecapStore

        # OpenRouter sessions of this run: <schema>:recap, then <schema>:<task session>.
        with llm_session(f"{schema}:recap"):
            manifest["recap"] = recap_scenario(store, case.scenario_id)
        dump(out / "recaps.json", RecapStore(store).export_scenario(case.scenario_id))
        from src.ledger.search import MessageIndex

        # Index the check-ins after their recaps, so a search key holds its recap items.
        manifest["search_index"] = MessageIndex(store).update(case.patient_id)
        if recap_only:
            manifest.update(recap_only=True, status="finished")
            return
        if seed_fhir:
            with tempfile.TemporaryDirectory() as directory:
                shutil.copy(case.fhir_path, directory)
                manifest["fhir_seed"] = seed(Path(directory), [case.patient_id])
        clinician = None
        if target is not None:
            # A note written by an earlier run would change this run's evidence and baseline.
            start = default_record_updates().fhir.read(target["resource"])
            text = note_text(start)
            if text != fixture_note(case, target["resource"]):
                raise RuntimeError(
                    f"{target['resource']} does not hold the fixture note; seed FHIR again."
                )
            manifest["record_target_start"] = {
                "version": resource_version(start),
                "text_sha256": content_sha256(text),
            }
            manifest["gate_judge"] = judge.identity()
            clinician = ScriptedClinician(
                case,
                default_record_updates(),
                RecordStore(schema=probe_schema),
                Judgments(out / "judgments", judge, judge.identity()),
            )
        records = execute_tasks(
            case,
            functools.partial(respond, condition=condition),
            clinician,
            session_prefix=schema,
        )
        if clinician is not None:
            dump(out / "records.json", clinician.records(judge.identity()))
        from src.imaging.results import ImagingResultStore

        dump(out / "imaging-results.json", ImagingResultStore(store).export())
        for record in records:
            dump(out / "tasks" / f"{record['question_id']}.json", record)
        manifest["tasks"] = {record["task_id"]: record["status"] for record in records}
        manifest["status"] = "finished"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        dump(out / "run.json", manifest)
        if not keep_ledger:
            for name in (schema, record_schema, probe_schema):
                _drop_schema(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", required=True, choices=["development", "test"])
    parser.add_argument("--scenarios", help="comma-separated scenario IDs (default: whole split)")
    parser.add_argument("--models", required=True, help="comma-separated OpenRouter model IDs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--confirm-test", action="store_true", help="required for the test split")
    parser.add_argument(
        "--no-fhir-seed", action="store_true", help="FHIR fixtures are already loaded"
    )
    parser.add_argument("--keep-ledger", action="store_true", help="keep the run's ledger schema")
    parser.add_argument(
        "--recap-only",
        action="store_true",
        help="import and recap the check-ins, then stop before the Doctor tasks",
    )
    parser.add_argument(
        "--condition",
        choices=["full", "full_ungoverned", "single_agent"],
        default="full",
        help="experiment condition: multi-agent system, without governance, or one agent",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if args.split == "test" and not args.confirm_test:
        parser.error("The test split is the manuscript experiment; add --confirm-test to run it.")
    release = Release.load()
    scenarios = args.scenarios.split(",") if args.scenarios else release.scenarios(args.split)
    models = args.models.split(",")
    cases = {scenario: release.case(args.split, scenario) for scenario in scenarios}

    if args.child:
        (model,), (scenario,) = models, scenarios
        run_case(
            cases[scenario],
            release,
            model,
            args.out,
            seed_fhir=not args.no_fhir_seed,
            keep_ledger=args.keep_ledger,
            recap_only=args.recap_only,
            condition=args.condition,
        )
        return 0

    failures = 0
    for model in models:
        for scenario in scenarios:
            out = args.out / model_slug(model) / scenario
            previous = out / "run.json"
            if previous.exists():
                if json.loads(previous.read_text()).get("status") == "finished":
                    print(f"skip {model} {scenario}: already finished", flush=True)
                    continue
                parser.error(
                    f"{out} holds an unfinished run; keep it for audit and use a new --out."
                )
            command = [
                sys.executable,
                "-m",
                "src.evaluation.run",
                "--child",
                "--split",
                args.split,
                "--scenarios",
                scenario,
                "--models",
                model,
                "--out",
                str(out),
            ]
            command += ["--confirm-test"] if args.confirm_test else []
            command += ["--no-fhir-seed"] if args.no_fhir_seed else []
            command += ["--keep-ledger"] if args.keep_ledger else []
            command += ["--recap-only"] if args.recap_only else []
            command += ["--condition", args.condition]
            print(f"run {model} {scenario}", flush=True)
            failures += subprocess.run(command, cwd=ROOT, check=False).returncode != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
