"""Task and data loading shared by label.py, train.py, evaluate.py and export_onnx.py.

A *task* (``task.json``) fixes the label set and the questions the model learns to answer:

    {
      "labels": ["small_model", "medium_model", "frontier_model"],
      "questions": [ <Laya question>, ... ],   # 1+ phrasings of the same decision
      "rubric": "..."                          # optional, used by label.py
    }

Each question is a regular Laya question. A ``choice`` question's ``criteria`` keys must be
exactly the labels (any order: options are shuffled in training). A ``score`` question's
``criteria`` list is ordinal and must have one entry per label, in label order.

*Data* is JSON Lines, one example per line:

    {"prompt": "hi!", "label": "small_model"}                       # one gold label
    {"prompt": "...", "labels": ["medium_model", "frontier_model"]}  # several annotators -> soft target
    {"state": {"ticket": "...", "plan": "pro"}, "label": "..."}      # any Laya state instead of a prompt
    {"prompt": "...", "label": "...", "split": "val"}                # optional fixed split

Rows labelled ``"exclude"`` (by any annotator) are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXCLUDE = "exclude"


class TaskError(ValueError):
    """Raised when task.json or a data file is malformed."""


def load_task(path: str | Path) -> dict[str, Any]:
    """Read and validate a task file; returns it with ``questions`` checked against ``labels``."""
    task = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = task.get("labels")
    if not isinstance(labels, list) or len(labels) < 2 or len(set(labels)) != len(labels):
        raise TaskError("'labels' must be a list of 2+ distinct strings")
    if EXCLUDE in labels:
        raise TaskError(f"'{EXCLUDE}' is reserved for skipping rows; rename that label")
    questions = task.get("questions")
    if not isinstance(questions, list) or not questions:
        raise TaskError("'questions' must be a non-empty list of Laya questions")
    for i, q in enumerate(questions):
        where = f"questions[{i}]"
        if q.get("type") == "choice":
            crit = q.get("criteria")
            keys = list(crit) if isinstance(crit, (dict, list)) else None
            if keys is None or set(keys) != set(labels) or len(keys) != len(labels):
                raise TaskError(f"{where}: a choice question's criteria keys must be exactly the labels")
        elif q.get("type") == "score":
            crit = q.get("criteria")
            if not isinstance(crit, list) or len(crit) != len(labels):
                raise TaskError(f"{where}: a score question needs one criterion per label, in label order")
        else:
            raise TaskError(f"{where}: type must be 'choice' or 'score'")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise TaskError(f"{where}: 'instructions' must be a non-empty string")
    return task


def row_state(row: dict[str, Any]) -> Any:
    """The Laya state for a data row: its ``state`` if given, else ``{"prompt": prompt}``."""
    if "state" in row:
        return row["state"]
    if "prompt" in row:
        return {"prompt": row["prompt"]}
    raise TaskError("each row needs a 'prompt' or a 'state'")


def row_votes(row: dict[str, Any]) -> list[str]:
    """All label votes on a row (``label`` and/or ``labels``)."""
    votes = list(row.get("labels") or [])
    if row.get("label") is not None:
        votes.append(row["label"])
    return votes


def load_rows(path: str | Path, labels: list[str], *, require_labels: bool = True) -> list[dict[str, Any]]:
    """Read a JSONL data file into rows with a soft ``target`` over ``labels``.

    The target is each label's share of the votes, so two annotators who disagree give a 50/50
    target. ``gold`` is the label when every vote agrees, else ``None`` (such rows still train,
    but accuracy is only reported on rows with a gold label).
    """
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskError(f"{path}:{n}: not valid JSON ({exc.msg})") from exc
        row_state(row)  # validates prompt/state presence
        row.setdefault("id", n)
        votes = row_votes(row)
        if EXCLUDE in votes:
            continue
        if not votes:
            if require_labels:
                raise TaskError(f"{path}:{n}: no 'label' or 'labels'")
            rows.append(row)
            continue
        unknown = sorted(set(votes) - set(labels))
        if unknown:
            raise TaskError(f"{path}:{n}: unknown label(s) {unknown}; expected one of {labels}")
        row["target"] = [votes.count(label) / len(votes) for label in labels]
        row["gold"] = votes[0] if len(set(votes)) == 1 else None
        rows.append(row)
    if not rows:
        raise TaskError(f"{path}: no usable rows")
    return rows


def answer_probs(answer: dict[str, Any], question: dict[str, Any], labels: list[str]) -> list[float]:
    """Map a Laya answer's probabilities back to label order (score options are ordinal)."""
    probs = answer.get("probabilities") or {}
    if question["type"] == "choice":
        return [float(probs.get(label, 0.0)) for label in labels]
    return [float(probs.get(str(i), probs.get(i, 0.0))) for i in range(len(labels))]
