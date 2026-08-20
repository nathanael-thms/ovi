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

# Define Package managers and commands
if command -v apt >/dev/null 2>&1; then
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt install -y"
    UPDATE_CMD="sudo apt update"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
    UPDATE_CMD=""  # dnf updates repositories automatically during install
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    INSTALL_CMD="sudo yum install -y"
    UPDATE_CMD=""
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
    echo "Checking system-level dependencies for OpenVINO GPU acceleration..."

    # Run the system update command
    if [ -n "$UPDATE_CMD" ]; then
        $UPDATE_CMD
    fi

    # Ensure OpenCL Core Loader is installed
    if ! ldconfig -p 2>/dev/null | grep -q "libOpenCL.so.1"; then
        echo "OpenCL loader missing. Installing opencl library..."
        case $PKG_MANAGER in
            apt)          $INSTALL_CMD ocl-icd-libopencl1 ;;
            dnf|yum)      $INSTALL_CMD ocl-icd ;;
            pacman)       $INSTALL_CMD ocl-icd ;;
        esac
    fi

    # Detect if Intel Graphics hardware capability is passed into the environment
    if [ -e /dev/dri/renderD128 ]; then
        echo "Graphics render nodes detected (/dev/dri/renderD128)."

        # Verify if the host environment uses Intel graphics drivers
        if lsmod 2>/dev/null | grep -qE "i915|xe"; then
            echo "Intel Graphics hardware modules (i915/xe) confirmed active."

            # Install Intel Compute Runtime drivers if vendor file is missing
            if [ ! -f /etc/OpenCL/vendors/intel.icd ] && [ ! -f /etc/OpenCL/vendors/intel-neo.icd ]; then
                echo "Intel Compute drivers missing. Installing..."
                case $PKG_MANAGER in
                    apt)          $INSTALL_CMD intel-opencl-icd ;;
                    dnf|yum)      $INSTALL_CMD intel-opencl ;;
                    pacman)       $INSTALL_CMD intel-compute-runtime ;;
                esac
            fi
        fi

        # Handle GPU Node Access Permissions dynamically
        if [ "$REAL_USER" != "root" ]; then
            echo "Ensuring user '$REAL_USER' has permissions for GPU hardware..."

            # Check which group owns the render node on this specific system
            RENDER_GROUP=$(ls -l /dev/dri/renderD128 2>/dev/null | awk '{print $4}' || echo "")

            # If the group is listed as 'root', force it to 'render' or 'video'
            if [ "$RENDER_GROUP" = "root" ]; then
                echo "Warning: /dev/dri/renderD128 group is incorrectly set to 'root'."
                if getent group render >/dev/null 2>&1; then
                    sudo chown root:render /dev/dri/renderD128
                    RENDER_GROUP="render"
                elif getent group video >/dev/null 2>&1; then
                    sudo chown root:video /dev/dri/renderD128
                    RENDER_GROUP="video"
                fi
                echo "Fixed hardware node ownership."
            fi

            for grp in video render "$RENDER_GROUP"; do
                if [ -n "$grp" ] && [ "$grp" != "root" ] && getent group "$grp" >/dev/null 2>&1; then
                    if ! groups "$REAL_USER" | grep -q "\b$grp\b"; then
                        echo "Adding $REAL_USER to $grp group..."
                        sudo usermod -aG "$grp" "$REAL_USER"
                        echo "--> Note: $REAL_USER may need to log out and back in for GPU group changes to apply."
                    fi
                fi
            done
        fi
    fi
else
    echo "Warning: Unknown or unsupported package manager. Skipping automated GPU driver checks."
fi

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

    if [ -n "$PKG_MANAGER" ]; then
        echo "Installing python venv package via $PKG_MANAGER..."
        case $PKG_MANAGER in
            apt)     $INSTALL_CMD "python${PY_VER}-venv" ;;
            dnf|yum) $INSTALL_CMD "python3" ;;
            pacman)  $INSTALL_CMD "python" ;; # Arch bundles venv inside the core package
        esac
    else
        echo "Error: Unknown package manager. Cannot automatically install python venv."
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