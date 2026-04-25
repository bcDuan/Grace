from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HEURISTICS = ("bm25", "sbert", "ppr")


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def by_qid(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("per_question", [])
    return {str(row.get("qid")): row for row in rows if row.get("qid") is not None}


def truncate(text: Any, limit: int = 80) -> str:
    s = str(text).replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


def has_turn_text(row: dict[str, Any]) -> bool:
    turns = row.get("retrieved_turns")
    return isinstance(turns, list) and any(isinstance(t, dict) and t.get("content") for t in turns)


def format_retrieved(row: dict[str, Any]) -> list[str]:
    if has_turn_text(row):
        lines = []
        for i, turn in enumerate(row.get("retrieved_turns", [])[:5], start=1):
            sid = turn.get("session_id", "unknown")
            role = turn.get("role", "unknown")
            content = truncate(turn.get("content", ""))
            lines.append(f"{i}. `{sid}` {role}: {content}")
        return lines

    ids = row.get("retrieved_turn_ids", [])[:5]
    sessions = row.get("retrieved_sessions", [])[:5]
    lines = []
    for i, tid in enumerate(ids, start=1):
        sess = sessions[i - 1] if i - 1 < len(sessions) else "unknown"
        lines.append(f"{i}. turn_id=`{tid}`, session=`{sess}`")
    return lines


def choose_best_heuristic(qid: str, heuristic_maps: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for name in HEURISTICS:
        row = heuristic_maps[name].get(qid)
        if row and bool(row.get("correct", False)):
            candidates.append((float(row.get("sess_recall_at_5", 0.0) or 0.0), name, row))
    if not candidates:
        raise ValueError(f"No correct heuristic row for qid={qid}")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, name, row = candidates[0]
    return name, row


def render_case(
    index: int,
    gnn_row: dict[str, Any],
    best_name: str,
    best_row: dict[str, Any],
    show_turn_warning: bool,
) -> str:
    lines = [
        f"## Case {index}: `{gnn_row.get('qid')}`",
        "",
        f"- **Question type:** `{gnn_row.get('question_type', 'unknown')}`",
        f"- **Question:** {gnn_row.get('question', '')}",
        f"- **Gold answer:** {gnn_row.get('gold', '')}",
        f"- **GNN predicted:** {gnn_row.get('predicted', '')}",
        f"- **GNN SessR@5:** {float(gnn_row.get('sess_recall_at_5', 0.0) or 0.0):.3f}",
        "",
        "### GNN Top-5 Retrieved Turns",
    ]
    if show_turn_warning:
        lines.append("> Warning: result JSON does not contain retrieved turn text; showing turn ids and sessions only.")
    lines.extend(format_retrieved(gnn_row))
    lines.extend(
        [
            "",
            f"### Best Heuristic: `{best_name}`",
            "",
            f"- **Predicted:** {best_row.get('predicted', '')}",
            f"- **SessR@5:** {float(best_row.get('sess_recall_at_5', 0.0) or 0.0):.3f}",
            "",
            f"### `{best_name}` Top-5 Retrieved Turns",
        ]
    )
    if show_turn_warning:
        lines.append("> Warning: result JSON does not contain retrieved turn text; showing turn ids and sessions only.")
    lines.extend(format_retrieved(best_row))
    lines.extend(
        [
            "",
            "### Diagnosis",
            "",
            "GNN retrieves turns that are [X] but miss [Y].",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract retrieval-generation-gap failure cases.")
    p.add_argument("--gnn", required=True)
    p.add_argument("--bm25", required=True)
    p.add_argument("--sbert", required=True)
    p.add_argument("--ppr", required=True)
    p.add_argument("--output", default="experiments/results/failure_cases.md")
    p.add_argument("--max-cases", type=int, default=5)
    args = p.parse_args()

    gnn = by_qid(load_result(args.gnn))
    heuristic_maps = {
        "bm25": by_qid(load_result(args.bm25)),
        "sbert": by_qid(load_result(args.sbert)),
        "ppr": by_qid(load_result(args.ppr)),
    }

    missing_turn_text = not any(has_turn_text(row) for row in gnn.values())
    if missing_turn_text:
        print(
            "[warn] No retrieved turn text found in result JSONs; "
            "failure_cases.md will show retrieved_turn_ids/retrieved_sessions only."
        )

    failures: list[tuple[float, str, dict[str, Any], str, dict[str, Any]]] = []
    by_type = Counter()
    saves = Counter()
    for qid, gnn_row in gnn.items():
        if bool(gnn_row.get("correct", False)):
            continue
        if float(gnn_row.get("sess_recall_at_5", 0.0) or 0.0) < 0.5:
            continue

        correct_heuristics = [
            name
            for name in HEURISTICS
            if qid in heuristic_maps[name] and bool(heuristic_maps[name][qid].get("correct", False))
        ]
        if not correct_heuristics:
            continue

        best_name, best_row = choose_best_heuristic(qid, heuristic_maps)
        sess_r = float(gnn_row.get("sess_recall_at_5", 0.0) or 0.0)
        failures.append((sess_r, qid, gnn_row, best_name, best_row))
        by_type[str(gnn_row.get("question_type", "unknown"))] += 1
        for name in correct_heuristics:
            saves[name] += 1

    failures.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = failures[: max(0, args.max_cases)]

    lines = [
        "# Appendix C Failure Cases",
        "",
        (
            "Selection criteria: GNN ep16 is incorrect, at least one heuristic retriever is correct, "
            "and GNN SessR@5 >= 0.5."
        ),
        "",
        f"- Total matched failure cases: **{len(failures)}**",
        f"- Cases shown: **{len(selected)}**",
        "",
    ]
    if missing_turn_text:
        lines.extend(
            [
                "> Warning: the input JSON files do not store retrieved turn text. "
                "This appendix shows retrieved turn ids and session ids instead.",
                "",
            ]
        )

    for i, (_, _, gnn_row, best_name, best_row) in enumerate(selected, start=1):
        lines.append(render_case(i, gnn_row, best_name, best_row, show_turn_warning=False))

    lines.extend(["## Summary", "", "### Failure Distribution by Question Type", ""])
    lines.append("| Question type | Count |")
    lines.append("|---|---:|")
    for qtype, count in sorted(by_type.items()):
        lines.append(f"| {qtype} | {count} |")

    lines.extend(["", "### Heuristic Saves", ""])
    lines.append("| Heuristic | Saves |")
    lines.append("|---|---:|")
    for name in HEURISTICS:
        lines.append(f"| {name} | {saves[name]} |")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}")
    print(f"[summary] matched={len(failures)} shown={len(selected)}")


if __name__ == "__main__":
    main()
