from pathlib import Path
from typing import Dict
from pypdf import PdfReader




def load_character_descriptions(roles_dir: Path) -> Dict[str, str]:
    """Load character descriptions from PDF files in agents/roles/*/description/"""
    descriptions = {}
    
    for role_dir in roles_dir.glob("*/description"):
        # Find PDF file in the description folder
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
    
    return descriptions