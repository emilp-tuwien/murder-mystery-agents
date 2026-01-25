from typing import List, Dict

def _banner(title: str, char: str = "=") -> None:
    width = max(60, len(title) + 12)
    line = char * width
    print("\n" + line)
    print(title.center(width))
    print(line)


def _section(title: str) -> None:
    width = max(50, len(title) + 10)
    line = "-" * width
    print("\n" + line)
    print(title.upper())
    print(line)


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(no conversation)"

    rows = [f"IDX | TURN | SPEAKER             | TEXT",
            f"----+------+---------------------+--------------------------------------------------"]
    for idx, u in enumerate(history, start=1):
        turn = u.get("turn", idx)
        speaker = u.get("speaker", "Unknown")[:21]
        text = u.get("text", "").strip()
        rows.append(f"{idx:03d} | T{turn:02d} | {speaker:<21} | {text}")
    return "\n".join(rows)