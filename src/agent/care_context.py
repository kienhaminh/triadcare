"""Check-in messages that a Care Agent answer cites, attached to the answer by code.

Care reads the stored check-in messages and cites their message IDs. Its statements are
a lossy rewrite; the messages are not. A consultation result carries the retrieved messages
next to the statements, so the Doctor and the answer check receive them as recorded.
"""

from __future__ import annotations

import re

CARE_AGENT = "Care Agent"


# The hyphens a model writes in IDs, U+2010 to U+2015 and the minus sign, one for one.
_HYPHENS = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"))


def plain(text: str) -> str:
    """The text with every hyphen written as "-"; positions do not change."""
    return text.translate(_HYPHENS)


# A full ledger ID, and one shortened with an ellipsis: the part after the scenario.
_FULL_ID = re.compile(r"(?:obs:)?imported:[\w.-]+:([\w.-]+(?::[\w.-]+)*)")
# A message of a live check-in, or an observation of one: no scenario to cut.
_LIVE_ID = re.compile(r"(?<![\w.:-])(?:obs:)?(live:[\w-]+(?::\d+)?)")
_SHORT_ID = re.compile(r"(?:\u2026|\.\.\.):?(?:obs:)?([\w.-]+(?::[\w.-]+){2,})")
_SEQUENCE = re.compile(r"\bseq-\d+\b")
# An ID written without its prefix: the check-in day and reporter, and the message or
# observation number.
_BARE_ID = re.compile(r"(?<![\w:.\u2026-])(day-\d+:[a-z]+:(?:seq-)?\d+)\b")
# More messages of the same check-in after an ID: "seq-2,4", "seq-2, seq-4", "and seq-4".
_MORE_SEQUENCES = re.compile(r"(?:\s*(?:,|and|&)\s*(?:seq-)?\d+\b)+")
# A template, not an ID: "day-N:patient:seq-M".
PLACEHOLDER = re.compile(r"(?:^|:)(?:day|seq)-[A-Za-z]+(?::|$)")


def references(text: str) -> list[tuple[str, int, bool]]:
    """(ID part after the scenario, position, shortened) for each ID the text names. A
    list of sequence numbers after an ID names those messages of its check-in. A check-in
    ID followed by "seq-N" before the next ID names that message."""
    text = plain(text)
    found = sorted(
        [(m.group(1), m.start(), m.end(1), False) for m in _FULL_ID.finditer(text)]
        + [(m.group(1), m.start(), m.end(1), True) for m in _SHORT_ID.finditer(text)]
        + [(m.group(1), m.start(), m.end(1), True) for m in _BARE_ID.finditer(text)]
        + [(m.group(1), m.start(), m.end(1), False) for m in _LIVE_ID.finditer(text)],
        key=lambda item: item[1],
    )
    result = []
    for index, (written, position, end, shortened) in enumerate(found):
        reference = written.rstrip(".-")
        if PLACEHOLDER.search(reference):
            continue
        result.append((reference, position, shortened))
        head, _, last = reference.rpartition(":")
        more = _MORE_SEQUENCES.match(text, end - len(written) + len(reference))
        if _SEQUENCE.fullmatch(last) and more:
            result.extend(
                (f"{head}:seq-{number}", position, shortened)
                for number in re.findall(r"\d+", more.group(0))
            )
        if reference.count(":") == 1 and not reference.startswith("live:"):
            # A check-in ID: day and reporter.
            stop = found[index + 1][1] if index + 1 < len(found) else len(text)
            result.extend(
                (f"{reference}:{m.group(0)}", m.start(), shortened)
                for m in _SEQUENCE.finditer(text, position, stop)
            )
    return result


def short_id(ledger_id: str) -> str:
    """The ID without "imported:<scenario>:". A live ID has no scenario and stays whole."""
    return ledger_id.split(":", 2)[2] if ledger_id.startswith("imported:") else ledger_id


def statement(row: dict) -> str:
    quote = f'"{row["quote"]}" (part of the message)' if row.get("quote_cut") else f'"{row["quote"]}"'
    context = row.get("question_context")
    if not isinstance(context, dict) or not isinstance(context.get("quote"), str):
        return quote
    question_id = context.get("message_id")
    reference = f" [{short_id(question_id)}]" if isinstance(question_id, str) else ""
    question = (
        f'"{context["quote"]}" (part of the message)'
        if context.get("quote_cut")
        else f'"{context["quote"]}"'
    )
    return (
        f"{quote}\n  Care Agent question (context only; not a reporter fact)"
        f"{reference}: {question}"
    )


def _check_in(row: dict) -> str:
    """The check-in of a row: its session ID, or for a live ID the ID without its last part."""
    return row.get("session_id") or row["source_message_id"].rsplit(":", 1)[0]


def _render(rows: list[dict], label) -> str:
    lines = []
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        sessions.setdefault(_check_in(row), []).append(row)
    for group in sessions.values():
        first = group[0]
        lines.append(f"{first['reporter']} check-in recorded {first['recorded_at']}:")
        lines.extend(f"- [{label(row)}] {statement(row)}" for row in group)
    return "\n".join(lines)


def render_observations(rows: list[dict]) -> str:
    """The retrieved messages as text: each check-in once, then each message with its ID and
    its words as recorded."""
    return _render(rows, lambda row: row["source_message_id"])


def written_text(result: dict) -> str:
    """The text Care wrote in a consultation result, without the attached records: the
    JSON fields as lines, the text of an invalid answer, or the prose of older runs."""
    if isinstance(result.get("answer"), str):
        return result["answer"]
    if result.get("status") == "invalid" and "text" in result:
        return result["text"] or ""
    lines = []
    for key, items in result.items():
        if key in _NOT_WRITTEN or not isinstance(items, list) or not items:
            continue
        lines.append(f"{key.replace('_', ' ').capitalize()}:")
        for item in items:
            if isinstance(item, dict) and "statement" in item:
                sources = ", ".join(item.get("sources") or [])
                lines.append(f"- {item.get('statement', '')} ({sources})")
            elif isinstance(item, dict):
                lines.append("- " + "; ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                lines.append(f"- {item}")
    return "\n".join(lines)


# Fields of a consultation result that its agent did not write.
_NOT_WRITTEN = frozenset(
    {
        "agent",
        "check_ins_read",
        "cited_observations",
        "cited_observations_cut",
        "problems",
        "removed",
    }
)


# The field of a Care result that maps each record ID ("c12") to its observation and
# ledger message. Model input leaves out fields that start with "_".
RECORDS_FIELD = "_records"
_LABEL = re.compile(r"\[(c\d+)\]")


def consultation_text(result: dict) -> str:
    """A consultation result as text: what Care wrote, then the retrieved rows when there are
    any. Rows are rendered text in a result the model read, and a list in older runs."""
    answer = written_text(result)
    rows = result.get("cited_observations") or []
    text = rows if isinstance(rows, str) else render_observations(rows)
    # Rows labeled for the Doctor ("c12") name their ledger message for other readers.
    labels = result.get(RECORDS_FIELD) or {}
    text = _LABEL.sub(
        lambda m: f"[{labels.get(m.group(1), {}).get('message_id', m.group(1))}]", text
    )
    return f"{answer}\n\nCited observations:\n{text}" if rows else answer
