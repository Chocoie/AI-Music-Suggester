import sys
from typing import List

# Core Python & Environment
try:
    import dotenv
    print("✅ python-dotenv: Installed successfully.")
except ImportError:
    print("❌ python-dotenv: Failed to install.")

# LangChain and Ollama
try:
    from langchain.chains import LLMChain
    from langchain_ollama.llms import OllamaLLM
    print("✅ LangChain Core: Installed successfully.")
    print("✅ LangChain Ollama: Installed successfully.")
except ImportError as e:
    print(f"❌ LangChain/Ollama: Failed to install ({e})")
    
# Spotify API
try:
    import spotipy
    print("✅ Spotipy (Spotify API): Installed successfully.")
except ImportError:
    print("❌ Spotipy (Spotify API): Failed to install.")

# Last.fm API
try:
    import pylast
    print("✅ Pylast (Last.fm API): Installed successfully.")
except ImportError:
    print("❌ Pylast (Last.fm API): Failed to install.")

# System Check
print("-" * 30)
print(f"Python Version: {sys.version.split()[0]}")