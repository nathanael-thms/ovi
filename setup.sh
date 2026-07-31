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
    
    # Keep sudo alive for as long as script runs
    while kill -0 "$$" 2>/dev/null; do sudo -n true; sleep 60; done &
fi

echo "Granting executable permissions to ovi"
chmod +x "$SCRIPT_DIR/ovi"
echo "Done"

# Change to script dir
cd "$SCRIPT_DIR"

# Install python3.14-venv if not present
sudo apt install -y python3.14-venv

# Create the venv and activate
python3 -m venv ovi-env
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