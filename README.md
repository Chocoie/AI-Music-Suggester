# Music Suggester Assistant

This project is a LangGraph-based conversational assistant that uses the Gemini API to route user queries to the Spotify Web API for music search.

## 1. Prerequisites

* Python 3.8+
* pip (Python package installer)

## 2. Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone "https://github.com/Chocoie/AI-Music-Suggester"
    cd AI-Music-Suggester
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r setup/requirements.txt
    ```

## 3. API Key Configuration

1.  **Gemini API Key:** Get a key from Google AI Studio.
2.  **Spotify API Credentials:**
    * Go to the Spotify Developer Dashboard.
    * Create an application to get a **Client ID** and **Client Secret**.
3.  **Create `.env` file:** Copy the contents of the provided `.env.example` file into a new file named **`.env`** in the root directory and fill in your keys.

    **.env (Example)**
    ```
    GOOGLE_API_KEY=<AIzaSy...your-key-here>
    SPOTIFY_CLIENT_ID=<...your-spotify-id...>
    SPOTIFY_CLIENT_SECRET=<...your-spotify-secret...>
    GRPC_VERBOSITY='ERROR'
    GLOG_minloglevel='2'    
    ```

## 4. Verify Setup
Execute the verification script to confirm all dependencies and environment variables are loaded correctly:
    ```bash
    # On macOS/Linux
    python3 setup/verify_install.py

    # On Windows
    python setup/verify_install.py
    ```
You must see a "✨ SUCCESS!" message before proceeding. If you see any "❌" messages, follow the troubleshooting steps provided by the script.

To check grpcio:
    ```
    pip show grpcio
    ```
You should see "Version: 1.67.1"

## 5. How to Run

Execute the main script:
    ```
    streamlit run app.py 
    ```