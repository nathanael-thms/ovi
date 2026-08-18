#!/bin/bash
set -euo pipefail

force=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      force=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for sudo privileges, if not, ask for authentication
if [ "$EUID" -ne 0 ]; then
    echo "Sudo access required. Please enter password:"
    if ! sudo -v; then
        echo "Error: Authentication failed. Aborting installation."
        exit 1
    fi
fi

echo "Granting executable permissions to ovi"
chmod +x "$SCRIPT_DIR/ovi"
echo "Done"

# Change to script dir
cd "$SCRIPT_DIR"

echo "Ensuring Python venv is available..."

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "Error: python3 not found."
    exit 1
fi

# Look for venv module, if it is not present, install it
if ! "$PYTHON_BIN" -m venv ovi-env 2>/dev/null; then
    echo "python3 venv module missing."

    # Detect Python minor version
    PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    # Install correct venv package based on manager type
    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y "python${PY_VER}-venv"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "python3"
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y "python3"
    else
        echo "Error: apt, dnf, and yum are not available. Cannot continue."
        exit 1
    fi

    # Retry venv creation (fatal if it fails)
    "$PYTHON_BIN" -m venv ovi-env
fi

# Activate the environment
source ovi-env/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "Installing global command(ovi)"
if [ "$force" = false ]; then
    if [ ! -f "/usr/local/bin/ovi" ] && ! command -v ovi &> /dev/null; then
        echo "Ovi not installed, proceeding"
    else
        echo "Warning: ovi already exists in /usr/local/bin or your PATH. Checking if this is an ovi installation."
        if [[ "$(ovi --is-ovi-install 2>/dev/null)" == "True" ]]; then
            echo "This is an OVI installation. Proceeding..."
        else
            echo "To prevent accidental command override, the script is aborting, please run: 'bash $SCRIPT_DIR/setup.sh --force' to proceed"
            exit 1
        fi
    fi
fi

echo "Creating global symlink..."
if ! sudo ln -sfn "$SCRIPT_DIR/ovi" /usr/local/bin/ovi; then
    echo "Error: Critical failure writing to /usr/local/bin/ovi even with sudo."
    exit 1
fi
echo "Setup completed successfully"