import platform
import subprocess
import os
import venv

VENV_NAME = ".venv"
REQUIREMENTS_FILE = "requirements.txt"
SCRIPT_TO_RUN = "app.py"


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

    # Create venv
    if not os.path.exists(VENV_NAME):
        create_venv(VENV_NAME)
    else:
        print(f"\n> Virtual environment '{VENV_NAME}' already exists. Using it.")

    # Install requirements
    install_requirements(VENV_NAME)

    # Run your Python application
    run_app(VENV_NAME)
