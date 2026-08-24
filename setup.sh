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

# Track the real, non-root user who invoked sudo
REAL_USER="${SUDO_USER:-$(whoami)}"

PKG_MANAGER=""
INSTALL_CMD=""
UPDATE_CMD=""
IS_RHEL_DERIVATIVE=false

# Define Package managers and commands
if command -v apt >/dev/null 2>&1; then
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt install -y"
    UPDATE_CMD="sudo apt update"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
    UPDATE_CMD=""
    # Check if we are specifically on an Enterprise Linux variant (Rocky, RHEL, AlmaLinux)
    if [ -f /etc/redhat-release ] && ! grep -qi "Fedora" /etc/redhat-release; then
        IS_RHEL_DERIVATIVE=true
    fi
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    INSTALL_CMD="sudo yum install -y"
    UPDATE_CMD=""
    if [ -f /etc/redhat-release ] && ! grep -qi "Fedora" /etc/redhat-release; then
        IS_RHEL_DERIVATIVE=true
    fi
elif command -v pacman >/dev/null 2>&1; then
    PKG_MANAGER="pacman"
    INSTALL_CMD="sudo pacman -S --noconfirm"
    UPDATE_CMD="sudo pacman -Sy"
fi

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

if [ -n "$PKG_MANAGER" ]; then
  if [ "$IS_RHEL_DERIVATIVE" = true ]; then
    echo "--> Warning: Automatic GPU acceleration dependencies installation not supported on enterprise RHEL enviroments"
  else
    echo "Checking system-level dependencies for OpenVINO GPU acceleration..."

    if [ -n "$UPDATE_CMD" ]; then
        $UPDATE_CMD
    fi

    # Install opencl library
    echo "Installing opencl library..."
    case $PKG_MANAGER in
        apt)
            $INSTALL_CMD ocl-icd-libopencl1 intel-opencl-icd
            ;;
        dnf|yum)
            $INSTALL_CMD ocl-icd intel-opencl
            ;;
        pacman)
            $INSTALL_CMD ocl-icd intel-compute-runtime
            ;;
    esac

    # Fix hardware device nodes permissions if they are passed into the environment
    if [ -e /dev/dri/renderD128 ]; then
        echo "Graphics render nodes detected (/dev/dri/renderD128). Configuring permissions..."

        # Explicitly force correct group ownership on primary card devices (e.g., card0, card1)
        for card in /dev/dri/card*; do
            if [ -e "$card" ]; then
                sudo chown root:video "$card"
                sudo chmod 660 "$card"
            fi
        done

        # Explicitly force correct group ownership on the compute render block
        if getent group render >/dev/null 2>&1; then
            sudo chown root:render /dev/dri/renderD128
        else
            sudo chown root:video /dev/dri/renderD128
        fi
        sudo chmod 660 /dev/dri/renderD128
        echo "Hardware node permission overrides applied successfully."

        if [ "$REAL_USER" != "root" ]; then
            echo "Ensuring user '$REAL_USER' has access to hardware groups..."
            for grp in video render; do
                if getent group "$grp" >/dev/null 2>&1; then
                    if ! groups "$REAL_USER" | grep -q "\b$grp\b"; then
                        echo "Adding $REAL_USER to $grp group..."
                        sudo usermod -aG "$grp" "$REAL_USER"
                        echo "--> Note: $REAL_USER may need to log out and back in for GPU group changes to apply."
                    fi
                fi
            done
        fi
    fi
  fi
else
    echo "Warning: Unknown or unsupported package manager. Skipping automated GPU driver checks."
fi

echo "Ensuring compatible Python environment is available..."

if [ "$IS_RHEL_DERIVATIVE" = true ] && [ "$PKG_MANAGER" = "dnf" ]; then
    if ! command -v python3.14 >/dev/null 2>&1; then
        echo "Installing modern python3.14 from AppStream..."
        $INSTALL_CMD python3.14 python3.14-pip
    fi
fi

# Track down the best available modern execution binary path
PYTHON_BIN=""
for bin in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$bin" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$bin")"
        echo "Selected Python engine binary: $bin"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: python3 not found. Cannot continue."
    exit 1
fi

# Look for venv module using our modern binary engine. If it is missing, install it
if ! "$PYTHON_BIN" -m venv ovi-env 2>/dev/null; then
    echo "Python venv module missing for selected engine."

    # Detect the minor version of our execution target
    PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [ -n "$PKG_MANAGER" ]; then
        echo "Installing matching python venv package via $PKG_MANAGER..."
        case $PKG_MANAGER in
            apt)     $INSTALL_CMD "python${PY_VER}-venv" ;;
            dnf|yum) $INSTALL_CMD "python${PY_VER}" ;;
            pacman)  $INSTALL_CMD "python" ;;
        esac
    else
        echo "Error: Unknown package manager. Cannot automatically install python venv."
        exit 1
    fi

    # Final retry to create the environment
    "$PYTHON_BIN" -m venv ovi-env
fi

# Activate the environment
source ovi-env/bin/activate

# Upgrade pip inside the venv
echo "Upgrading virtual environment package tools (pip)..."
python3 -m pip install --upgrade pip setuptools wheel

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
