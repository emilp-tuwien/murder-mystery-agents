
import subprocess
from typing import List, Optional

def _get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models on the system."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output - skip header line and extract model names
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return []
        
        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if parts:
                # Model name is the first column
                model_name = parts[0]
                models.append(model_name)
        
        return sorted(models)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch Ollama models: {e}")
        return []
    except FileNotFoundError:
        print("Warning: 'ollama' command not found. Please install Ollama.")
        return []
    except Exception as e:
        print(f"Warning: Error fetching Ollama models: {e}")
        return []


def _select_ollama_model() -> Optional[str]:
    """Let user select from available Ollama models."""
    models = _get_available_ollama_models()
    
    if not models:
        print("No Ollama models found. Please install models using 'ollama pull <model_name>'")
        return None
    
    print("\nAvailable Ollama models:")
    for idx, model in enumerate(models, start=1):
        print(f"  {idx}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None