#!/bin/bash
set -euo pipefail

force=false
verbose=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      force=true
      shift
      ;;
    --verbose)
      verbose=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [ -f /etc/alpine-release ] || [ -d /etc/apk ] || grep -qi "alpine" /etc/os-release 2>/dev/null; then
    echo "================================================================="
    echo "ERROR: Alpine Linux is not supported by this installer."
    echo "--> OpenVINO requires a glibc-based environment (e.g. openSUSE, Ubuntu)."
    echo "================================================================="
    exit 1
fi


# Setup background tool redirection based on verbose flag
if [ "$verbose" = true ]; then
    exec {TOOL_OUT}>&1
    exec {TOOL_ERR}>&2
else
    exec {TOOL_OUT}>/dev/null
    exec {TOOL_ERR}>/dev/null
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Track the real, non-root user who invoked sudo
REAL_USER="${SUDO_USER:-$(whoami)}"

PKG_MANAGER=""
INSTALL_CMD=""
UPDATE_CMD=""
IS_RHEL_DERIVATIVE=false

# Define Package managers and commands with silent flags built-in
if command -v apt >/dev/null 2>&1; then
    PKG_MANAGER="apt"
    if [ "$verbose" = true ]; then
        INSTALL_CMD="sudo DEBIAN_FRONTEND=noninteractive apt-get install -y"
        UPDATE_CMD="sudo apt-get update"
    else
        INSTALL_CMD="sudo DEBIAN_FRONTEND=noninteractive apt-get -qq -y install"
        UPDATE_CMD="sudo apt-get -qq update"
    fi
elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    if [ "$verbose" = true ]; then
        INSTALL_CMD="sudo dnf install -y"
    else
        INSTALL_CMD="sudo dnf install -y -q"
    fi
    UPDATE_CMD=""
    if [ -f /etc/redhat-release ] && ! grep -qi "Fedora" /etc/redhat-release; then
        IS_RHEL_DERIVATIVE=true
    fi
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    if [ "$verbose" = true ]; then
        INSTALL_CMD="sudo yum install -y"
    else
        INSTALL_CMD="sudo yum install -y -q"
    fi
    UPDATE_CMD=""
    if [ -f /etc/redhat-release ] && ! grep -qi "Fedora" /etc/redhat-release; then
        IS_RHEL_DERIVATIVE=true
    fi
elif command -v pacman >/dev/null 2>&1; then
    PKG_MANAGER="pacman"
    if [ "$verbose" = true ]; then
        INSTALL_CMD="sudo pacman -S --noconfirm"
        UPDATE_CMD="sudo pacman -Sy"
    else
        INSTALL_CMD="sudo pacman -S --noconfirm --quiet"
        UPDATE_CMD="sudo pacman -Sy --quiet"
    fi
elif command -v zypper >/dev/null 2>&1; then
    PKG_MANAGER="zypper"
    if [ "$verbose" = true ]; then
        INSTALL_CMD="sudo zypper --non-interactive install -y"
        UPDATE_CMD="sudo zypper refresh"
    else
        INSTALL_CMD="sudo zypper --quiet --non-interactive install -y"
        UPDATE_CMD="sudo zypper --quiet refresh"
    fi
fi

# Check for sudo privileges, if not, ask for authentication
if [ "$EUID" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
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
        $UPDATE_CMD >&$TOOL_OUT 2>&$TOOL_ERR
    fi

    # Install opencl library
    echo "Installing opencl library..."
    case $PKG_MANAGER in
        apt)
            $INSTALL_CMD ocl-icd-libopencl1 intel-opencl-icd >&$TOOL_OUT 2>&$TOOL_ERR
            ;;
        dnf|yum)
            $INSTALL_CMD ocl-icd intel-opencl >&$TOOL_OUT 2>&$TOOL_ERR
            ;;
        pacman)
            $INSTALL_CMD ocl-icd intel-compute-runtime >&$TOOL_OUT 2>&$TOOL_ERR
            ;;
        zypper)
            $INSTALL_CMD intel-opencl libOpenCL1 >&$TOOL_OUT 2>&$TOOL_ERR
            ;;
    esac

    # Fix hardware device nodes permissions if they are passed into the environment
    if [ -e /dev/dri/renderD128 ]; then
        echo "Graphics render nodes detected (/dev/dri/renderD128). Configuring permissions..."

        # Explicitly force correct group ownership on primary card devices (e.g., card0, card1)
        for card in /dev/dri/card*; do
            if [ -e "$card" ]; then
                if command -v sudo >/dev/null 2>&1; then
                    sudo chown root:video "$card"
                    sudo chmod 660 "$card"
                else
                    chown root:video "$card"
                    chmod 660 "$card"
                fi
            fi
        done

        # Explicitly force correct group ownership on the compute render block
        SUDO_PREFIX="sudo"
        if ! command -v sudo >/dev/null 2>&1; then SUDO_PREFIX=""; fi

        if getent group render >/dev/null 2>&1; then
            $SUDO_PREFIX chown root:render /dev/dri/renderD128
        else
            $SUDO_PREFIX chown root:video /dev/dri/renderD128
        fi
        $SUDO_PREFIX chmod 660 /dev/dri/renderD128
        echo "Hardware node permission overrides applied successfully."

        # Fallback path search for distributions like openSUSE
        USERMOD_BIN="usermod"
        if ! command -v usermod >/dev/null 2>&1 && [ -x /usr/sbin/usermod ]; then
            USERMOD_BIN="/usr/sbin/usermod"
        fi

        if [ "$USERMOD_BIN" != "usermod" ] || command -v usermod >/dev/null 2>&1; then
            echo "Ensuring user '$REAL_USER' has access to hardware groups..."
            for grp in video render; do
                if getent group "$grp" >/dev/null 2>&1; then
                    if ! groups "$REAL_USER" | grep -q "\b$grp\b"; then
                        echo "Adding $REAL_USER to $grp group..."
                        $SUDO_PREFIX $USERMOD_BIN -aG "$grp" "$REAL_USER"
                        echo "--> Note: $REAL_USER may need to log out and back in for GPU group changes to apply."
                    fi
                fi
            done
        elif [ "$REAL_USER" != "root" ] && command -v addgroup >/dev/null 2>&1; then
             echo "Ensuring user '$REAL_USER' has access to hardware groups..."
             for grp in video render; do
                if getent group "$grp" >/dev/null 2>&1; then
                    if ! groups "$REAL_USER" | grep -q "\b$grp\b"; then
                        echo "Adding $REAL_USER to $grp group..."
                        $SUDO_PREFIX addgroup "$REAL_USER" "$grp"
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

HAS_MODERN_PYTHON=false
for ver in 3.14 3.13 3.12 3.11; do
    if command -v "python${ver}" >/dev/null 2>&1; then
        HAS_MODERN_PYTHON=true
       break
    fi
done

# Only force-install a modern package if the system is completely lacking one
if [ "$HAS_MODERN_PYTHON" = false ]; then
    echo "Installing modern python3.14..."
    case $PKG_MANAGER in
        apt)     $INSTALL_CMD "python3.14" >&$TOOL_OUT 2>&$TOOL_ERR ;;
        dnf|yum) $INSTALL_CMD "python3.14" >&$TOOL_OUT 2>&$TOOL_ERR ;;
        pacman)  $INSTALL_CMD "python" >&$TOOL_OUT 2>&$TOOL_ERR ;;
    esac
fi

# For Zypper
if [ "$PKG_MANAGER" = "zypper" ]; then
     if command -v "python3" >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_VERSION_FOR_ZYPPER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
        if [[ ! "$PYTHON_VERSION" =~ ^3\.(11|12|13|14)$ ]]; then
            $INSTALL_CMD "python314" "python3" "python314-pip" "python314-curses" >&$TOOL_OUT 2>&$TOOL_ERR
            echo "You may need to run ovi via python3.14 command rather than simply typing ovi, see docs for details"
        else
            $INSTALL_CMD "python${PYTHON_VERSION_FOR_ZYPPER}-pip" "python${PYTHON_VERSION_FOR_ZYPPER}-curses" >&$TOOL_OUT 2>&$TOOL_ERR
        fi
    else
      $INSTALL_CMD "python3" >&$TOOL_OUT 2>&$TOOL_ERR
      PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_VERSION_FOR_ZYPPER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
        if [[ ! "$PYTHON_VERSION" =~ ^3\.(11|12|13|14)$ ]]; then
            $INSTALL_CMD "python314" "python3" "python314-pip" "python314-curses" >&$TOOL_OUT 2>&$TOOL_ERR
            echo "You may need to run ovi via python3.14 command rather than simply typing ovi, see docs for details"
        else
            $INSTALL_CMD "python${PYTHON_VERSION_FOR_ZYPPER}-pip" "python${PYTHON_VERSION_FOR_ZYPPER}-curses" >&$TOOL_OUT 2>&$TOOL_ERR
        fi
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
if ! "$PYTHON_BIN" -m venv ovi-env >/dev/null 2>&1; then
    echo "Python venv module missing for selected engine."

    # Detect the minor version of our execution target
    PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [ -n "$PKG_MANAGER" ]; then
        echo "Installing matching python venv package via $PKG_MANAGER..."
        case $PKG_MANAGER in
            apt)     $INSTALL_CMD "python${PY_VER}-venv" >&$TOOL_OUT 2>&$TOOL_ERR ;;
            dnf|yum) $INSTALL_CMD "python${PY_VER}" >&$TOOL_OUT 2>&$TOOL_ERR ;;
            pacman)  $INSTALL_CMD "python" >&$TOOL_OUT 2>&$TOOL_ERR ;;
        esac
    else
        echo "Error: Unknown package manager. Cannot automatically install python venv."
        exit 1
    fi

    # Final retry to create the environment
    "$PYTHON_BIN" -m venv ovi-env >/dev/null 2>&1
fi

# Activate the environment
source ovi-env/bin/activate

# Upgrade pip inside the venv
echo "Upgrading virtual environment package tools (pip)..."
if [ "$verbose" = true ]; then
    python3 -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
else
    python3 -m pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1
    pip install -q -r requirements.txt >/dev/null 2>&1
fi

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
