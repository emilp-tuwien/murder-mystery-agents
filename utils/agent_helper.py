from pathlib import Path
from typing import Dict, Tuple
from pypdf import PdfReader


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
            text = "\n".join([page.extract_text() for page in pdf.pages])
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
                # Convert folder name to character name (e.g., "michael-nightshade" -> "Michael Nightshade")
                character_name = role_dir.name.replace("-", " ").title()
                # Minimal base persona - all knowledge comes from rounds
                descriptions[character_name] = f"You are {character_name}."
    
    return descriptions


def load_round_description(roles_dir: Path, character_name: str, round_num: int) -> str:
    """Load round-specific description for a character."""
    # Convert character name to folder format (e.g., "Michael Nightshade" -> "michael-nightshade")
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
            # Check if this folder matches the character name
            dir_name_normalized = role_dir.name.lower().replace("-", " ").replace("'", "'")
            char_name_normalized = character_name.lower()
            if dir_name_normalized == char_name_normalized:
                alt_path = role_dir / "rounds" / str(round_num) / "description.txt"
                if alt_path.exists():
                    try:
                        return alt_path.read_text().strip()
                    except Exception as e:
                        print(f"Error loading round {round_num} description for {character_name}: {e}")
    return ""


def load_confession(roles_dir: Path, character_name: str) -> str:
    """Load confession text for a character."""
    # Convert character name to folder format
    folder_name = character_name.lower().replace(" ", "-")
    
    confession_path = roles_dir / folder_name / "confession.txt"
    
    if confession_path.exists():
        try:
            return confession_path.read_text().strip()
        except Exception as e:
            print(f"Error loading confession for {character_name}: {e}")
            return ""
    
    # Try alternative folder names if not found
    for role_dir in roles_dir.iterdir():
        if role_dir.is_dir() and not role_dir.name.startswith("_"):
            dir_name_normalized = role_dir.name.lower().replace("-", " ").replace("'", "'")
            char_name_normalized = character_name.lower()
            if dir_name_normalized == char_name_normalized:
                alt_path = role_dir / "confession.txt"
                if alt_path.exists():
                    try:
                        return alt_path.read_text().strip()
                    except Exception as e:
                        print(f"Error loading confession for {character_name}: {e}")
    return ""


def detect_murderer(roles_dir: Path, character_name: str) -> bool:
    """
    Detect if a character is the murderer by checking their round 1 description.
    The murderer's description will contain indication that they committed the murder.
    """
    round1_desc = load_round_description(roles_dir, character_name, 1)
    confession = load_confession(roles_dir, character_name)
    
    # Check for explicit murderer indicator
    combined_text = (round1_desc + " " + confession).lower()
    
    # Look for explicit "YOU ARE THE MURDERER" statement
    if "you are the murderer" in combined_text:
        return True
    
    return False