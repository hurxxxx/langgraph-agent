"""
Check if environment variables are being loaded correctly.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API keys are loaded
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY", "Not found"))
print("SERPER_API_KEY:", os.getenv("SERPER_API_KEY", "Not found"))
print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY", "Not found"))
print("ANTHROPIC_API_KEY:", os.getenv("ANTHROPIC_API_KEY", "Not found"))
