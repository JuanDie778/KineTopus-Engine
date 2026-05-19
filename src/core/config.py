
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Load environment variables from .env file
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path)

print(f"DEBUG: Loading .env from {env_path}")
print(f"DEBUG: GEMINI_API_KEY available: {bool(os.getenv('GEMINI_API_KEY'))}")

class Config:
    """Central configuration management."""
    pass
