from typing import Dict, List, Optional

# Default outer width for all drawn boxes (border chars included).
BOX_WIDTH = 70


def _boxed(text, width: int = BOX_WIDTH, style: str = "double") -> str:
    """Return a perfectly aligned box around one or more centered lines.

    text:  a single string or a list of strings (one per body line).
    style: "double" (╔═╗) for headline banners, "single" (┌─┐) for sub-boxes.

    All box drawing in the game goes through here so borders always line up
    regardless of the title length — no more hand-counted spaces.
    """
    styles = {
        "double": ("╔", "╗", "╚", "╝", "═", "║"),
        "single": ("┌", "┐", "└", "┘", "─", "│"),
    }
    tl, tr, bl, br, h, v = styles.get(style, styles["double"])
    inner = max(2, width - 2)
    lines = text if isinstance(text, list) else [text]
    top = tl + h * inner + tr
    bottom = bl + h * inner + br
    body = [f"{v}{str(line).center(inner)}{v}" for line in lines]
    return "\n".join([top, *body, bottom])


def _banner(title: str, char: str = "=") -> None:
    width = max(BOX_WIDTH, len(title) + 12)
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


def _wrap(text: str, width: int) -> List[str]:
    """Greedy word-wrap; always returns at least one (possibly empty) line."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(no conversation)"

    TEXT_W = 64
    NAME_W = 20
    header = f"{'IDX':>3} | {'TURN':>4} | {'RND':>3} | {'SPEAKER':<{NAME_W}} | TEXT"
    sep = f"{'-'*3}-+-{'-'*4}-+-{'-'*3}-+-{'-'*NAME_W}-+-{'-'*TEXT_W}"
    rows = [header, sep]
    for idx, u in enumerate(history, start=1):
        turn = int(u.get("turn", idx))
        rnd = u.get("round", "")
        speaker = str(u.get("speaker", "Unknown"))[:NAME_W]
        text = (u.get("text", "") or "").strip()
        wrapped = _wrap(text, TEXT_W)
        rows.append(f"{idx:03d} | T{turn:>3} | {str(rnd):>3} | {speaker:<{NAME_W}} | {wrapped[0]}")
        for cont in wrapped[1:]:
            rows.append(f"{'':>3} | {'':>4} | {'':>3} | {'':<{NAME_W}} | {cont}")
    return "\n".join(rows)


def _format_suspicion_matrix(
    assessments: Dict[str, object],
    all_agents: List[str],
    murderer_name: Optional[str] = None,
    round_num: Optional[int] = None,
) -> str:
    """Render a suspicion heatmap: rows = observers, columns = suspects.

    ``assessments`` maps an observer's name to its RoundSuspicionAssessment
    (with ``.suspect_assessments``, ``.top_suspect``, ``.overall_uncertainty``).
    Cells hold that observer's 1–10 suspicion_score for the column suspect;
    the diagonal (observer == suspect) shows '—'. The actual murderer's column
    is flagged with '*' so the reader can see at a glance whether the group is
    converging on the right person.
    """
    observers = [a for a in all_agents if a in assessments]
    if not observers:
        return ""

    COL = 10
    NAME_W = 16

    def col_label(name: str) -> str:
        # Use the first name so the header stays readable instead of mid-word
        # truncations like "Margar"/"Paulin". The murderer is flagged with '*'.
        label = name.split()[0]
        if murderer_name and name == murderer_name:
            label += "*"
        return label[: COL - 1]

    title = f"SUSPICION MATRIX — Round {round_num}" if round_num is not None else "SUSPICION MATRIX"
    header = f"{'observer / suspect':<{NAME_W}}" + "".join(f"{col_label(s):>{COL}}" for s in all_agents)
    lines = [title, header, "-" * len(header)]

    for obs in observers:
        assessment = assessments[obs]
        scores = {sa.suspect: sa.suspicion_score for sa in assessment.suspect_assessments}
        cells = ""
        for suspect in all_agents:
            if suspect == obs:
                cells += f"{'—':>{COL}}"
            else:
                value = scores.get(suspect)
                cells += f"{(str(value) if value is not None else '·'):>{COL}}"
        top = getattr(assessment, "top_suspect", "?")
        unc = getattr(assessment, "overall_uncertainty", "?")
        suffix = f"  ◀ top: {top} (unc {unc}/10)"
        lines.append(f"{obs[:NAME_W]:<{NAME_W}}{cells}{suffix}")

    lines.append("scores: 1 (innocent) – 10 (guilty)   '*' = actual murderer   '—' = self   '·' = no score")
    key = "   ".join(
        f"{a.split()[0]} = {a}" + ("*" if murderer_name and a == murderer_name else "")
        for a in all_agents
    )
    lines.append(f"key: {key}")
    return "\n".join(lines)
