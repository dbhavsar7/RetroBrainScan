import platform
import subprocess
import os
import venv
import sqlite3

VENV_NAME = ".venv"
REQUIREMENTS_FILE = "requirements.txt"
SCRIPT_TO_RUN = "app.py"
IMAGE_FOLDER = "OUTPUT_IMAGES"
DB_NAME = "RetroBrainScanDB.db"


# -------------------------------
# Database Utilities
# -------------------------------
def init_database(db_path=DB_NAME):
    """Initialize SQLite3 database with ImageData table"""
    print("\n=========== Initializing Database ===========")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create ImageData table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ImageData (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_data TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Create AnalysisResults table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS AnalysisResults (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                patient_info TEXT,
                current_risk_score REAL,
                current_prediction TEXT,
                future_risk_score REAL,
                future_prediction TEXT,
                current_regions TEXT,
                future_regions TEXT,
                care_plan TEXT,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES ImageData(id)
            )
        ''')
        
        conn.commit()
        print(f"> Database '{db_path}' initialized successfully.")
        print("> ImageData table created or already exists.")
        print("> AnalysisResults table created or already exists.")
        conn.close()
        
    except sqlite3.Error as e:
        print(f"ERROR: Database initialization failed - {e}")
        return False
    
    return True


# -------------------------------
# Virtual Environment Utilities
# -------------------------------
def get_venv_paths(venv_dir):
    system = platform.system()

    if system == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
        activate_path = os.path.join(venv_dir, "Scripts", "activate")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")
        activate_path = os.path.join(venv_dir, "bin", "activate")

    return pip_path, python_path, activate_path


def create_venv(venv_dir=VENV_NAME):
    print("\n=========== Creating Virtual Environment ===========")
    venv.create(venv_dir, with_pip=True)
    print(f"> Virtual environment created at: {venv_dir}")


# -------------------------------
# Install requirements.txt
# -------------------------------
def install_requirements(venv_dir=VENV_NAME):
    pip_path, python_path, activate_path = get_venv_paths(venv_dir)

    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"\nERROR: requirements.txt not found in this directory.")
        return

    print("\n=========== Installing Requirements ===========")
    cmd = [
        pip_path,
        "install",
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
        "-r", REQUIREMENTS_FILE
    ]

    subprocess.run(cmd)
    print("\n> Requirements installed successfully.")


# --- Function to create a folder if it does not exist ---
def createFolderIfNotExists(folder_path=IMAGE_FOLDER):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"> Folder '{folder_path}' created successfully.")
    else:
        print(f"> Folder '{folder_path}' already exists.")


# --- Function to setup .env file if it doesn't exist ---
def setup_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    env_file = ".env"
    env_example = ".env.example"
    
    if os.path.exists(env_file):
        print(f"> .env file already exists.")
        return
    
    if os.path.exists(env_example):
        print(f"\n=========== Setting up .env file ===========")
        print(f"> Copying {env_example} to {env_file}...")
        print(f"> ⚠️  IMPORTANT: Please edit {env_file} and add your GOOGLE_AI_API_KEY")
        try:
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print(f"> {env_file} created. Please add your API key.")
        except Exception as e:
            print(f"> Warning: Could not create {env_file}: {e}")
    else:
        print(f"\n=========== Environment Configuration ===========")
        print(f"> ⚠️  No .env file found. Creating template...")
        env_template = """# Google Gemini AI Configuration
# Get your API key from: https://makersuite.google.com/app/apikey

GOOGLE_AI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
"""
        try:
            with open(env_file, 'w') as f:
                f.write(env_template)
            print(f"> {env_file} created. Please add your GOOGLE_AI_API_KEY")
        except Exception as e:
            print(f"> Warning: Could not create {env_file}: {e}")

# -------------------------------
# Run the app.py file
# -------------------------------
def run_app(venv_dir=VENV_NAME):
    pip_path, python_path, activate_path = get_venv_paths(venv_dir)

    print("\n=========== Running App ===========")
    subprocess.run([python_path, SCRIPT_TO_RUN])


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    createFolderIfNotExists()
    
    # Setup .env file
    setup_env_file()

    # Initialize database
    init_database(DB_NAME)

    # Create venv
    if not os.path.exists(VENV_NAME):
        create_venv(VENV_NAME)
    else:
        print(f"\n> Virtual environment '{VENV_NAME}' already exists. Using it.")

    # Install requirements
    install_requirements(VENV_NAME)

    # Run your Python application
    run_app(VENV_NAME)
