import sys
import os
from dotenv import load_dotenv

# Check Python Version
MIN_PYTHON_VERSION = (3, 8)
if sys.version_info < MIN_PYTHON_VERSION:
    print(f"❌ ERROR: Your Python version ({sys.version_info.major}.{sys.version_info.minor}) is too old.")
    print(f"This script requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher.")
    sys.exit(1)


# Check Dependencies (Imports)

REQUIRED_LIBRARIES = [
    'langchain_core',
    'langgraph',
    'langchain_google_genai',
    'spotipy',
    'google.cloud.secretmanager',
    'dotenv',
    'streamlit'
]

print("--- 🔬 Checking Required Libraries ---")
all_libs_ok = True
for lib in REQUIRED_LIBRARIES:
    try:
        if '.' in lib:
            # For complex modules like google.cloud.secretmanager, import the top level
            __import__(lib.split('.')[0])
        else:
            __import__(lib)
        print(f"✅ Found {lib}")
    except ImportError:
        print(f"❌ Missing library: {lib}")
        all_libs_ok = False

if not all_libs_ok:
    print("\n🚨 Please run 'pip install -r setup/requirements.txt' to fix missing libraries.")
    sys.exit(1)


# Check .env File and Environment Variables
print("\n--- 🔑 Checking API Keys and Environment (.env file) ---")

# Load variables from .env file. This is the crucial step.
if not load_dotenv(override=True):
    print("⚠️ WARNING: Could not find a .env file. Please ensure it exists in the root directory.")

# List of essential environment variables
REQUIRED_ENV_VARS = [
    "GOOGLE_API_KEY",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET"
]

all_env_ok = True
for var in REQUIRED_ENV_VARS:
    value = os.getenv(var)
    
    if not value:
        print(f"❌ Missing environment variable: {var}")
        print(f"   Ensure you have created a '.env' file and filled in the value for {var}.")
        all_env_ok = False
    elif "YOUR_" in value:
        print(f"⚠️ Unfilled placeholder for: {var}")
        print(f"   Please replace the placeholder value in your '.env' file with your actual key.")
        all_env_ok = False
    else:
        print(f"✅ Found and loaded {var}")

# Optional check for GCP_PROJECT_ID if Secret Manager is intended for use
if os.getenv("GCP_PROJECT_ID"):
    print("ℹ️ Note: GCP_PROJECT_ID is set. Ensure your application has IAM permissions for Secret Manager.")

if not all_env_ok:
    print("\n🚨 Environment variable check failed. Please check your .env file.")
    sys.exit(1)

# Success Message
print("\n✨ SUCCESS! All required libraries and API keys are correctly configured.")
print("You are now ready to run the Music Suggester Assistant.")
