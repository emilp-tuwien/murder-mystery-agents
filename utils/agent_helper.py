from pathlib import Path
from typing import Dict, Tuple
import re
from pypdf import PdfReader


def _extract_display_name_from_round1(round1_path: Path, fallback: str) -> str:
    if not round1_path.exists():
        return fallback
    try:
        text = round1_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return fallback

    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if re.fullmatch(r"[A-Za-z .'-]+", candidate) and len(candidate.split()) >= 2:
            return candidate
    return fallback


def load_character_descriptions(roles_dir: Path) -> Dict[str, str]:
    """
    Load character names and base descriptions. Returns minimal base persona.
    Round-specific knowledge is loaded separately via update_round() to allow
    knowledge accumulation across rounds.
    """
    descriptions = {}
    
    # First try loading from PDF files (original method) - these contain base character info
    for role_dir in roles_dir.glob("*/description"):
        pdf_files = list(role_dir.glob("*.pdf"))
        if not pdf_files:
            continue
        
        pdf_path = pdf_files[0]
        character_name = pdf_path.stem  # filename without extension
        
        try:
            pdf = PdfReader(str(pdf_path))
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            descriptions[character_name.replace("-", " ").title()] = text
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
    
    # If no PDFs found, just return character names with minimal base persona
    # Round-specific knowledge will be loaded via update_round()
    if not descriptions:
        for role_dir in roles_dir.iterdir():
            if not role_dir.is_dir():
                continue
            
            # Skip __pycache__ and other non-role directories
            if role_dir.name.startswith("_"):
                continue
            
            # Check if this role has round descriptions
            rounds_dir = role_dir / "rounds"
            if rounds_dir.exists() and rounds_dir.is_dir():
                fallback_name = role_dir.name.replace("-", " ").title()
                round1_path = role_dir / "rounds" / "1" / "description.txt"
                character_name = _extract_display_name_from_round1(round1_path, fallback_name)
                # Minimal base persona - all knowledge comes from rounds
                descriptions[character_name] = f"You are {character_name}."
    
    return descriptions


def load_round_description(roles_dir: Path, character_name: str, round_num: int) -> str:
    """Load round-specific description for a character."""
    # Convert character name to folder format (e.g., "Bobby Herrerra" -> "bobby-herrerra")
    # Handle special characters like apostrophes
    folder_name = character_name.lower().replace(" ", "-")
    
    description_path = roles_dir / folder_name / "rounds" / str(round_num) / "description.txt"
    
    if description_path.exists():
        try:
            return description_path.read_text().strip()
        except Exception as e:
            print(f"Error loading round {round_num} description for {character_name}: {e}")
            return ""
    
    # Try alternative folder names if not found
    for role_dir in roles_dir.iterdir():
        if role_dir.is_dir() and not role_dir.name.startswith("_"):
            # Check if this folder matches the character name. Normalize the curly
            # apostrophe (U+2019) that often appears in PDF-extracted names so it
            # matches ASCII apostrophes used in folder paths (e.g. "O'Brien").
            dir_name_normalized = role_dir.name.lower().replace("-", " ").replace("’", "'")
            char_name_normalized = character_name.lower().replace("’", "'")
            if dir_name_normalized == char_name_normalized:
                alt_path = role_dir / "rounds" / str(round_num) / "description.txt"
                if alt_path.exists():
                    try:
                        return alt_path.read_text().strip()
                    except Exception as e:
                        print(f"Error loading round {round_num} description for {character_name}: {e}")
    return ""


def _load_role_file(roles_dir: Path, character_name: str, filename: str) -> str:
    folder_name = character_name.lower().replace(" ", "-")
    candidate_path = roles_dir / folder_name / filename

    if candidate_path.exists():
        try:
            return candidate_path.read_text().strip()
        except Exception as e:
            print(f"Error loading {filename} for {character_name}: {e}")
            return ""

    for role_dir in roles_dir.iterdir():
        if role_dir.is_dir() and not role_dir.name.startswith("_"):
            dir_name_normalized = role_dir.name.lower().replace("-", " ").replace("’", "'")
            char_name_normalized = character_name.lower().replace("’", "'")
            if dir_name_normalized == char_name_normalized:
                alt_path = role_dir / filename
                if alt_path.exists():
                    try:
                        return alt_path.read_text().strip()
                    except Exception as e:
                        print(f"Error loading {filename} for {character_name}: {e}")
    return ""


def load_confession(roles_dir: Path, character_name: str) -> str:
    """Load confession text for a character."""
    return _load_role_file(roles_dir, character_name, "confession.txt")


def load_murderer_strategy(roles_dir: Path, character_name: str) -> str:
    """Load optional murderer strategy guidance for a character."""
    return _load_role_file(roles_dir, character_name, "murderer_strategy.md")


def load_known_facts(roles_dir: Path, character_name: str) -> str:
    """Load the character's atomic first-hand fact list, if authored.

    ``known_facts.txt`` is an explicit, first-person enumeration of exactly what
    this character personally saw/knows. It is injected high-salience into the
    speaking/thinking prompts so the model states grounded testimony instead of
    paraphrasing (and distorting) the prose brief. Optional — returns "" if the
    scenario has not authored one for this role."""
    return _load_role_file(roles_dir, character_name, "known_facts.txt")


def detect_murderer(roles_dir: Path, character_name: str) -> bool:
    """
    Detect if a character is the murderer from their authored role files.
    The murderer's ``known_facts.txt`` is explicitly marked "(THE MURDERER)";
    the round 1 briefing may also carry an explicit indicator.
    """
    round1_desc = load_round_description(roles_dir, character_name, 1)
    known_facts = load_known_facts(roles_dir, character_name)

    # Check for an explicit murderer indicator in the authored role files.
    combined_text = (round1_desc + " " + known_facts).lower()

    if "you are the murderer" in combined_text or "(the murderer)" in combined_text:
        return True

    return False