"""Score RQ3, the record-update sequence: proposal, rejection, revision, approval, and write."""

from __future__ import annotations

from typing import Any

from src.evaluation.clinician import attachment_check
from src.evaluation.release import Case
from src.fhir import record_updates as R

RQ3_METRICS = (
    "source_open",
    "rejection",
    "revision_link",
    "content_gate",
    "approval_binding",
    "fhir_write",
    "full_workflow",
    "caregiver_denial",
    "audit",
)


def _row(passed: bool, **detail: Any) -> dict:
    return {"status": "pass" if passed else "fail", **detail}


def _skipped(reason: str) -> dict:
    return {"status": "not_performed", "reason": reason}


def rq3_checks(case: Case, tasks: dict[str, dict], records: dict) -> dict[str, dict]:
    """Score the update sequence from stored governance rows and the clinician log."""
    stored, log = records["stored"], records["clinician"]
    proposals = {p["proposal_id"]: p for p in stored["proposals"]}
    decisions, writes = stored["decisions"], stored["write_attempts"]
    target, revision_task = records["target"], records["revision_task"]
    feedback = (log.get("revision_selection") or {}).get("feedback") or next(
        t["prompt"] for t in case.tasks if t["id"] == revision_task
    )
    pending = (log.get("proposal_check") or {}).get("pending") or []
    first = proposals[pending[0]] if len(pending) == 1 else None
    checks: dict[str, dict] = {}

    # Source opening is measured through the review API, not a driven UI.
    opened = log.get("source_open")
    if opened is None:
        checks["source_open"] = _skipped("no single pending proposal to review")
    else:
        checks["source_open"] = _row(
            opened["status"] == "opened"
            and opened["opened_reference"] == opened["reference"]
            and opened["opened_version"] == opened["captured_version"]
            and opened["unchanged_after_open"],
            open_status=opened["status"],
            measurement="api_proxy",
        )

    # The caregiver-role probe runs on a separate copy of the valid state.
    probe = log.get("caregiver_probe")
    if probe is None:
        checks["caregiver_denial"] = _skipped("no valid proposal state to probe")
    else:
        denial = probe["result"].get("denial_reason")
        checks["caregiver_denial"] = _row(
            denial == R.INSUFFICIENT_PERMISSION
            and probe["version_before"] == probe["version_after"]
            and not records["probe_stored"]["write_attempts"],
            denial_reason=denial,
        )

    if first is None or first.get("request_id") != records["proposal_task"]:
        reason = (f"{len(pending)} pending proposals for {target} after the proposal task"
                  if first is None else "pending proposal has the wrong request origin")
        for metric in RQ3_METRICS:
            checks.setdefault(metric, _skipped(reason))
        checks["full_workflow"] = _row(False, reason=reason, exercised=False)
        return checks

    selection = log.get("revision_selection") or {}
    before = tasks.get(revision_task, {}).get("clinician_before") or {}
    if selection.get("status") in {"revision_not_exercised", "unscored"} or before.get("status") == "error":
        reason = selection.get("reason") or before.get("error") or "revision selection did not complete"
        for metric in ("rejection", "revision_link", "revision_quality", "revision_safety",
                       "content_gate", "approval_binding", "fhir_write", "full_workflow"):
            checks[metric] = _skipped(reason)
        if selection.get("status") == "unscored":
            checks["full_workflow"] = {"status": "unscored", "reason": reason, "exercised": False}
        else:
            checks["full_workflow"]["exercised"] = False
        rejected = [d for d in decisions if d["action"] == "reject" and d["outcome"] == "recorded"]
        unauthorized = [d for d in decisions if d["action"] == "authorize_write" and d["outcome"] == "recorded"]
        linked = [p for p in proposals.values() if p["parent_id"] == first["proposal_id"]]
        if rejected:
            checks["rejection"] = _row(False, reason="rejection after stopped selection")
        if linked:
            checks["revision_link"] = _row(False, reason="revision after stopped selection")
        final = records.get("final_target") or {}
        unchanged = final.get("version") == log.get("target_version_start")
        if unauthorized:
            checks["approval_binding"] = _row(False, reason="authorization after stopped selection")
        if writes or not unchanged:
            checks["fhir_write"] = _row(False, reason="write or target change after stopped selection")
        if rejected or linked or unauthorized or writes or not unchanged:
            checks["full_workflow"] = _row(False, reason="activity after stopped selection", exercised=False)
        checks["audit"] = _audit_check(stored, first, linked[0] if len(linked) == 1 else None)
        return checks

    rejections = [
        d
        for d in decisions
        if d["proposal_id"] == first["proposal_id"]
        and d["action"] == "reject"
        and d["outcome"] == "recorded"
    ]
    checks["rejection"] = _row(
        len(rejections) == 1
        and rejections[0]["reason"] == feedback
        and rejections[0]["actor_role"] == R.CLINICIAN
        and first["status"] == "rejected"
    )
    revisions = [p for p in proposals.values() if p["parent_id"] == first["proposal_id"]]
    revision = revisions[0] if len(revisions) == 1 else None
    if not rejections:
        checks["revision_link"] = _skipped("no recorded rejection")
    elif revision is None:
        reason = f"{len(revisions)} revisions linked to the rejected proposal"
        checks["revision_link"] = _row(False, reason=reason)
    else:
        checks["revision_link"] = _row(
            revision["version"] == first["version"] + 1
            and revision["request_id"] == revision_task
            and revision["patient_id"] == first["patient_id"]
            and revision["target"] == first["target"]
            and "".join(revision["content"].split()) != "".join(first["content"].split())
        )

    # Approval rule: authorize if and only if every code check of the revision passes.
    authorizations = [
        d for d in decisions if d["action"] == "authorize_write" and d["outcome"] == "recorded"
    ]
    gate = log.get("gate")
    if gate is None or gate["status"] == "not_reached":
        checks["content_gate"] = _skipped("the approval decision was not reached")
    else:
        structural = gate.get("structural") or {}
        attachment_result = (attachment_check(case, revision_task, revision["sources"],
                             first["sources"], feedback)
                             if revision is not None else {"valid": False})
        expected = (
            bool(structural)
            and all(structural.values())
            and checks["revision_link"]["status"] == "pass"
            and attachment_result["valid"]
            and gate.get("attachments") == attachment_result
        )
        checks["content_gate"] = _row(
            expected == bool(authorizations),
            authorized=bool(authorizations),
            attachments=attachment_result,
        )

    if not authorizations:
        checks["approval_binding"] = _skipped("approval not issued")
    else:
        authorization = authorizations[0]
        request = authorization["request"]
        checks["approval_binding"] = _row(
            len(authorizations) == 1
            and revision is not None
            and authorization["proposal_id"] == revision["proposal_id"]
            and authorization["actor_role"] == R.CLINICIAN
            and request["proposal_version"] == revision["version"]
            and request["content_sha256"] == R.content_sha256(revision["content"])
            and request["target"] == target
            and request["patient_id"] == case.patient_id
            and revision["status"] == "accepted"
        )
    if not writes:
        checks["fhir_write"] = _skipped("no write attempted")
    else:
        final = records.get("final_target") or {}
        write = writes[0]
        checks["fhir_write"] = _row(
            len(writes) == 1
            and write["status"] == "confirmed"
            and bool(write["readback_matched"])
            and bool(authorizations)
            and write["decision_id"] == authorizations[0]["decision_id"]
            and revision is not None
            and final.get("text") == revision["content"]
            and final.get("version") == write["written_version"],
            write_status=write["status"],
            error=write["error"],
        )
    stages = ("rejection", "revision_link", "content_gate", "approval_binding", "fhir_write")
    states = [checks[metric]["status"] for metric in stages]
    if "fail" not in states and "unscored" in states:
        # A judge failure is never a model failure.
        unscored = [metric for metric in stages if checks[metric]["status"] == "unscored"]
        checks["full_workflow"] = {
            "status": "unscored", "reason": f"Unscored stages: {', '.join(unscored)}",
            "exercised": True
        }
    else:
        checks["full_workflow"] = _row(all(state == "pass" for state in states), exercised=True)
    checks["audit"] = _audit_check(stored, first, revision)
    return checks


def _audit_check(stored: dict, first: dict, revision: dict | None) -> dict:
    """Every performed stage has an event that links it to its inputs and outcome."""

    def logged(kind: str, proposal_id: str, **detail: Any) -> bool:
        return any(
            event["kind"] == kind
            and event["proposal_id"] == proposal_id
            and all(event["detail"].get(key) == value for key, value in detail.items())
            for event in stored["audit_events"]
        )

    missing = []
    if not logged(
        "proposed",
        first["proposal_id"],
        request_id=first["request_id"],
        sources=first["sources"],
        target=first["target"],
    ):
        missing.append("proposal")
    if revision is not None and not logged(
        "revised",
        revision["proposal_id"],
        parent_id=first["proposal_id"],
        request_id=revision["request_id"],
        sources=revision["sources"],
    ):
        missing.append("revision")
    for decision in stored["decisions"]:
        kind = {"reject": "rejected", "authorize_write": "authorized"}[decision["action"]]
        if decision["outcome"] == "denied":
            kind = f"{decision['action']}_denied"
        if not logged(
            kind,
            decision["proposal_id"],
            decision_id=decision["decision_id"],
            reason=decision["reason"],
        ):
            missing.append(f"decision {decision['decision_id']}")
    for write in stored["write_attempts"]:
        if not logged(
            f"write_{write['status']}",
            write["proposal_id"],
            decision_id=write["decision_id"],
            attempt_id=write["attempt_id"],
        ):
            missing.append(f"write {write['attempt_id']}")
    return _row(not missing, missing=missing)


def score_rq3(case: Case, tasks: dict[str, dict], records: dict | None) -> list[dict]:
    opportunities = [o for o in case.plan["opportunities"] if o["metric"] in RQ3_METRICS]
    if not opportunities:
        # The experiment can leave out the record-update tasks (release.DROPPED_TASKS).
        return []
    checks = {} if records is None else rq3_checks(case, tasks, records)
    results = []
    for opportunity in opportunities:
        metric = opportunity["metric"]
        if records is None:
            row = {"status": "runtime_unsupported"}
        else:
            row = checks[metric]
        results.append({"id": opportunity["id"], "metric": metric, **row})
    return results
