"""Search of check-in messages by meaning: embedding retrieval, then an LLM rerank.

Each patient or caregiver message has two search keys. The message key is a context line
(recording date, weekday, reporter, speaker), the message text, and the recap items that
cite the message. The question key adds the Care Agent question before the message, so a
short answer ("I honestly don't know") matches its topic. A search takes the CANDIDATES
nearest messages of each key and merges them; a fixed rerank model orders the merged list
by how directly each message answers the query.

Development units, one topic query per unit (hit@1 / hit@3):
- embedding of the message key: 0.54 / 0.74;
- with rerank: 0.67 / 0.87;
- with both keys and rerank: 0.71 / 0.88. The top match was on topic for all units.
A full-text rank fused with the embedding rank lowered hit@1 to 0.38, so the search has
no full-text part.
"""

import re

import httpx

from src.config._env import env_str
from src.ledger.store import LedgerStore
from src.ledger.values import recorded_weekday

# Experiment settings of the search.
EMBEDDING_DIMENSIONS = 1536
CANDIDATES = 20  # Nearest messages of each key that the rerank orders.
_QUESTION_KEY = "+question"  # Model suffix of the question key.
RERANK_MODEL = "openai/gpt-6-luna"  # Fixed for all conditions; RERANK_MODEL overrides it.
BATCH = 64
TIMEOUT_SECONDS = 60.0


def _embedding_model() -> tuple[str, str, str]:
    base_url = env_str("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = env_str("EMBEDDING_MODEL", "text-embedding-3-small")
    key = env_str("EMBEDDING_API_KEY")
    if not key:
        raise ValueError("Set EMBEDDING_API_KEY to search check-ins by meaning.")
    return base_url, model, key


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches, in input order."""
    base_url, model, key = _embedding_model()
    vectors: list[list[float]] = []
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        for start in range(0, len(texts), BATCH):
            response = client.post(
                f"{base_url}/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "input": texts[start : start + BATCH]},
            )
            if response.status_code != 200:
                raise ValueError(f"The embedding service returned HTTP {response.status_code}.")
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
    return vectors


def _literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in vector) + "]"


def _key_text(message: dict, recap_items: list[str]) -> str:
    day = message["recorded_at"][:10]
    head = (
        f"{day} ({recorded_weekday(message['recorded_at'])}), {message['audience']} check-in, "
        f"{message['speaker']}: {message['content']}"
    )
    return head + ("\nRecap: " + "; ".join(recap_items) if recap_items else "")


def _question_key(message: dict, recap_items: list[str]) -> str:
    asked = f"Asked: {message['asked']}\n" if message.get("asked") else ""
    return asked + _key_text(message, recap_items)


_ASKED = """(SELECT q.content FROM messages AS q
             WHERE q.session_id = m.session_id AND q.sequence < m.sequence
               AND q.speaker = 'care_agent'
             ORDER BY q.sequence DESC LIMIT 1) AS asked"""


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Candidates ordered by the rerank model. On a failed or unclear answer, the
    embedding order stays."""
    lines = "\n".join(
        f"[{i}] {row['recorded_at']} {row['speaker']} (asked: {(row['asked'] or '')[:120]}): "
        f"{row['content'][:400]}"
        for i, row in enumerate(candidates)
    )
    prompt = (
        f"Query: {query}\n\nRank the messages by how directly they answer the query. Reply "
        f"with the indices, most relevant first, comma-separated, all {len(candidates)} of "
        f"them, nothing else.\n\n{lines}"
    )
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {env_str('OPENROUTER_API_KEY')}"},
            json={
                "model": env_str("RERANK_MODEL", RERANK_MODEL),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return candidates
    order = [int(i) for i in re.findall(r"\d+", answer) if int(i) < len(candidates)]
    order = list(dict.fromkeys(order)) + [i for i in range(len(candidates)) if i not in order]
    return [candidates[i] for i in order]


class MessageIndex:
    """The message_embeddings table in the ledger schema."""

    def __init__(self, store: LedgerStore) -> None:
        self.store = store
        self.model = _embedding_model()[1]
        with store._connect() as connection:
            store.lock_schema_ddl(connection)
            # The vector extension lives in the public schema; the ledger connection
            # searches only the ledger schema, so the type and operator are qualified.
            connection.execute(
                f"""CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id TEXT NOT NULL REFERENCES messages(message_id),
                    model TEXT NOT NULL,
                    key_text TEXT NOT NULL,
                    embedding public.vector({EMBEDDING_DIMENSIONS}) NOT NULL,
                    PRIMARY KEY (message_id, model)
                )"""
            )

    def _recap_items(self, patient_id: str) -> dict[str, list[str]]:
        """Recap change and unknown texts by the message_id that each item cites."""
        items: dict[str, list[str]] = {}
        with self.store._connect() as connection:
            exists = connection.execute("SELECT to_regclass('recaps') AS name").fetchone()
            if exists["name"] is None:
                return items
            rows = connection.execute(
                "SELECT changes, unknowns FROM recaps WHERE patient_id = %s AND status = 'stored'",
                (patient_id,),
            ).fetchall()
        for row in rows:
            for item in [*row["changes"], *row["unknowns"]]:
                if item.get("message_id") and item.get("text"):
                    items.setdefault(item["message_id"], []).append(item["text"])
        return items

    def update(self, patient_id: str) -> int:
        """Embed both keys of the patient's messages that have none yet; return how many
        messages."""
        with self.store._connect() as connection:
            rows = connection.execute(
                LedgerStore._message_select().replace("s.source_day", "s.source_day, " + _ASKED)
                + """ WHERE s.patient_id = %s AND m.speaker IN ('patient', 'caregiver')
                      AND NOT EXISTS (SELECT 1 FROM message_embeddings AS e
                                      WHERE e.message_id = m.message_id AND e.model = %s)
                      ORDER BY m.recorded_at, m.sequence""",
                (patient_id, self.model),
            ).fetchall()
        if not rows:
            return 0
        recap_items = self._recap_items(patient_id)
        keys = [
            (model, row, build(row, recap_items.get(row["message_id"], [])))
            for model, build in (
                (self.model + _QUESTION_KEY, _question_key),
                (self.model, _key_text),  # Written last: it marks the message as indexed.
            )
            for row in rows
        ]
        vectors = embed([key for _, _, key in keys])
        with self.store._connect() as connection:
            for (model, row, key), vector in zip(keys, vectors, strict=True):
                connection.execute(
                    """INSERT INTO message_embeddings (message_id, model, key_text, embedding)
                       VALUES (%s, %s, %s, %s::public.vector) ON CONFLICT DO NOTHING""",
                    (row["message_id"], model, key, _literal(vector)),
                )
        return len(rows)

    def nearest(self, patient_id: str, query: str, **scope) -> list[dict]:
        """The CANDIDATES scoped messages nearest to the query by each key, merged in
        alternating order, each with the Care Agent question before it."""
        self.update(patient_id)
        vector = _literal(embed([query])[0])
        where, parameters = LedgerStore._where(patient_id, **scope)
        lists = []
        with self.store._connect() as connection:
            for model in (self.model, self.model + _QUESTION_KEY):
                lists.append(
                    connection.execute(
                        f"""SELECT m.message_id, m.session_id, m.recorded_at, m.speaker,
                                m.content, {_ASKED}
                            FROM messages AS m
                            JOIN sessions AS s ON s.session_id = m.session_id
                            JOIN message_embeddings AS e
                              ON e.message_id = m.message_id AND e.model = %s
                            WHERE {where}
                            ORDER BY e.embedding OPERATOR(public.<=>) %s::public.vector,
                                m.message_id
                            LIMIT {CANDIDATES}""",
                        [model, *parameters, vector],
                    ).fetchall()
                )
        merged = {}
        for pair in zip(*lists, strict=False):
            for row in pair:
                merged.setdefault(row["message_id"], row)
        for row in [*lists[0], *lists[1]]:
            merged.setdefault(row["message_id"], row)
        return list(merged.values())

    def search(self, patient_id: str, query: str, *, limit: int, **scope) -> list[dict]:
        """The best matches, best first: message_id and session_id."""
        ranked = rerank(query, self.nearest(patient_id, query, **scope))
        return [{k: row[k] for k in ("message_id", "session_id")} for row in ranked[:limit]]
