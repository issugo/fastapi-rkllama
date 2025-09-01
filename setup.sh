#!/bin/bash

set -euo pipefail

function failed {
  echo "${1}" >&2
  exit $2
}

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Determine script location to find application root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
APP_ROOT="${SCRIPT_DIR}"
CONFIG_DIR="${APP_ROOT}/config"

# Create config directory if it doesn't exist
mkdir -p "${CONFIG_DIR}"

if test -z "${USE_UV}" -o -z "${USE_CONDA}" ; then
  USE_UV=true
  UV_ARG="" # This will hold "--no-uv" if uv is disabled
  USE_CONDA=false
  CONDA_ARG="--no-conda" # This will hold "--no-conda" if conda is disabled
  for arg in "$@"; do
      if [[ "$arg" == "--no-uv" ]]; then
          USE_UV=false
          UV_ARG="--no-uv"
          echo -e "${YELLOW}Uv is disabled for this installation.${RESET}"
      elif [[ "$arg" == "--no-conda" ]]; then
          USE_CONDA=false
          CONDA_ARG="--no-conda"
          echo -e "${YELLOW}Miniconda is disabled for this installation.${RESET}"
      elif [[ "$arg" == "--uv" ]]; then
          USE_UV=true
          UV_ARG="" # This will hold "--no-uv" if uv is disabled
      elif [[ "$arg" == "--conda" ]]; then
          USE_CONDA=true
          CONDA_ARG="" # This will hold "--no-conda" if conda is disabled
      fi
  done
fi


if test -z "${UV_DIR}" ; then
  # Uv installation path
  UV_DIR=~/miniconda3
fi
UV_URL="https://astral.sh/uv/install.sh"

if test -z "${MINICONDA_DIR}" ; then
  # Miniconda installation path
  MINICONDA_DIR=~/miniconda3
fi
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"

# Install uv ro Miniconda (if enabled)
if $USE_UV; then
  if [ -x "${UV_DIR}/uv" ]; then
    echo -e "${GREEN}uv is already installed.${RESET}"
    uv self update
  else
    echo -e "${YELLOW}uv is not installed. Proceeding with installation...${RESET}"
    curl -LsSf "${UV_URL}" | sh
    echo -e "${GREEN}uv was successfully installed.${RESET}"
  fi
elif $USE_CONDA; then
  if [ -d "$MINICONDA_DIR" ]; then
    echo -e "${GREEN}Miniconda is already installed.${RESET}"
  else
    echo -e "${YELLOW}Miniconda is not installed. Proceeding with installation...${RESET}"
    wget "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
    echo -e "${GREEN}Miniconda was successfully installed.${RESET}"
  fi
fi

# Check for Git repository updates
echo -e "${CYAN}Checking for updates...${RESET}"
git pull || true
echo -e "${GREEN}Update check completed successfully!${RESET}"

# Create the RKLLAMA directory in user's home
INSTALL_DIR="$HOME/RKLLAMA"
echo -e "${CYAN}Installing RKLLAMA to $INSTALL_DIR...${RESET}"
mkdir -p "$INSTALL_DIR"
echo -e "${CYAN}Copying resources to $INSTALL_DIR...${RESET}"
cp -rf . "$INSTALL_DIR/"

# Generate initial configuration first so we can use it to create the right directories
echo -e "${CYAN}Generating initial configuration...${RESET}"

cd "${INSTALL_DIR}" || failed "## failed to cd ${INSTALL_DIR}" 1

if test ! -s requirements.in -a -s requirements.txt ; then
  # create dummy requirements.in
  cat <<EOF | tee ./requirements-dev.in
requests
huggingface_hub
python-dotenv
transformers
torch
pydantic
fastapi
uvicorn
EOF
fi
if test -s requirements.in -a ! -s requirements-dev.in ; then
  cat <<EOF | tee ./requirements-dev.in
-r requirements.in
-c requirements.txt

ruff
pytest
pytest-asyncio
httpx
pre-commit
EOF
fi

if $USE_UV; then
  # install python
  uv python install

  test ! -s pyproject.toml && \
    uv init

  uv run - <<EOF
import config
config.validate()
EOF
else
  python3 -c "import config; config.validate()"
fi

# Create required directories based on configured paths
echo -e "${CYAN}Creating required directories based on configuration...${RESET}"
source "$INSTALL_DIR/config/config.env"
mkdir -p "$RKLLAMA_PATHS_MODELS_RESOLVED"
mkdir -p "$RKLLAMA_PATHS_LOGS_RESOLVED"
mkdir -p "$RKLLAMA_PATHS_DATA_RESOLVED"
mkdir -p "$RKLLAMA_PATHS_TEMP_RESOLVED"
mkdir -p "$RKLLAMA_PATHS_SRC_RESOLVED"
mkdir -p "$RKLLAMA_PATHS_LIB_RESOLVED"

function install_reqs {
  PIP=$1
  UV=$2
  if test -s requirements.in ; then
    ${UV} ${PIP} compile requirements.in -o requirements.txt
    test -n "${UV}" && ${UV} add -r requirements.in -c requirements.txt
  fi
  if test -s requirements-dev.in ; then
    ${UV} ${PIP} compile requirements-dev.in -o requirements-dev.txt
  fi

  test ! -s requirements.txt && failed "## cannot find requirements.txt from $(pwd)" 1
  # Install dependencies using pip
  echo -e "${CYAN}Installing dependencies from requirements.txt...${RESET}"
  ${UV} ${PIP} install -r requirements.txt

  if test -s requirements-dev.txt ; then
      # Install dependencies-dev using pip
      echo -e "${CYAN}Installing dependencies from requirements-dev.txt...${RESET}"
      ${UV} ${PIP} install -r requirements-dev.txt
      test -n "${UV}" && (
        if grep '^-r ' requirements-dev.in 1>/dev/null ; then
          sed '/^-r /d' requirements-dev.in | ${UV} add --dev -r - -c requirements-dev.txt
        else
          uv add --dev -r requirements-dev.in -c requirements-dev.txt
        fi
      )
  fi
}

# Install python libraries
echo -e "\e[32m=======Installing Python dependencies=======\e[0m"
# Add flask-cors to the pip install command
pip install requests flask huggingface_hub flask-cors python-dotenv transformers


for PYTHON_ENV in venv .venv ; do
  grep -e "^${PYTHON_ENV}/" ./.gitignore || ( echo "${PYTHON_ENV}/" | tee ./.gitignore 1>/dev/null )
done

if $USE_UV; then
  uv venv
  install_reqs pip uv
elif $USE_CONDA; then
  # Activate Miniconda and install dependencies (if enabled)
  source "${MINICONDA_DIR}/bin/activate"
  install_reqs pip3
elif test -d .venv ; then
  source .venv/bin/activate
  install_reqs pip
elif test -d venv ; then
  source venv/bin/activate
  install_reqs pip
else
  install_reqs pip3
fi



# Make client.sh and server.sh executable
echo -e "${CYAN}Making scripts executable${RESET}"
chmod +x "$INSTALL_DIR/client.sh"
chmod +x "$INSTALL_DIR/server.sh"
chmod +x "$INSTALL_DIR/uninstall.sh"

# Modify client.sh and server.sh to always use --no-conda if conda is disabled
if $USE_UV && ! $USE_CONDA ; then
  echo -e "${CYAN}Ensuring client.sh and server.sh always run with --uv --no-conda${RESET}"

  # Add --no-uv to client.sh if not already present
  if ! grep -q -- "--no-uv" "$INSTALL_DIR/client.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --uv --no-conda "$@"|' "$INSTALL_DIR/client.sh"
  fi

  # Add --no-conda to server.sh if not already present
  if ! grep -q -- "--no-conda" "$INSTALL_DIR/server.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --uv --no-conda "$@"|' "$INSTALL_DIR/server.sh"
  fi
elif ! $USE_UV && $USE_CONDA ; then
  echo -e "${CYAN}Ensuring client.sh and server.sh always run with --conda --no-uv${RESET}"

  # Add --no-uv to client.sh if not already present
  if ! grep -q -- "--no-uv" "$INSTALL_DIR/client.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --conda --no-uv "$@"|' "$INSTALL_DIR/client.sh"
  fi

  # Add --no-conda to server.sh if not already present
  if ! grep -q -- "--no-conda" "$INSTALL_DIR/server.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --conda --no-uv "$@"|' "$INSTALL_DIR/server.sh"
  fi
elif ! $USE_UV; then
  echo -e "${CYAN}Ensuring client.sh and server.sh always run with --no-uv${RESET}"

  # Add --no-uv to client.sh if not already present
  if ! grep -q -- "--no-uv" "$INSTALL_DIR/client.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --no-uv "$@"|' "$INSTALL_DIR/client.sh"
  fi

  # Add --no-conda to server.sh if not already present
  if ! grep -q -- "--no-conda" "$INSTALL_DIR/server.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --no-uv "$@"|' "$INSTALL_DIR/server.sh"
  fi
elif ! $USE_CONDA; then
  echo -e "${CYAN}Ensuring client.sh and server.sh always run with --no-conda${RESET}"

  # Add --no-conda to client.sh if not already present
  if ! grep -q -- "--no-conda" "$INSTALL_DIR/client.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --no-conda "$@"|' "$INSTALL_DIR/client.sh"
  fi

  # Add --no-conda to server.sh if not already present
  if ! grep -q -- "--no-conda" "$INSTALL_DIR/server.sh"; then
    sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --no-conda "$@"|' "$INSTALL_DIR/server.sh"
  fi
fi

# Create a global executable for rkllama that properly handles arguments
echo -e "${CYAN}Creating a global executable for rkllama...${RESET}"

cat <<'EOF' | sudo tee /usr/local/bin/rkllama >/dev/null
#!/bin/bash

# Use user's installation directory
INSTALL_DIR="$HOME/RKLLAMA"
CONFIG_DIR="$INSTALL_DIR/config"

# Source configuration if available
if [ -f "$CONFIG_DIR/config.env" ]; then
    source "$CONFIG_DIR/config.env"
fi

# Parse arguments to pass along
ARGS=""
PORT_ARG=""
USE_UV={{USE_UV}}
USE_CONDA={{USE_CONDA}}

for arg in "$@"; do
    if [[ "$arg" == "serve" ]]; then
        # Special handling for 'serve' command
        COMMAND="serve"
    elif [[ "$arg" == "--no-conda" ]]; then
        # Handle no-conda flag
        USE_CONDA=false
    elif [[ "$arg" == "--no-uv" ]]; then
        # Handle no-uv flag
        USE_UV=false
    elif [[ "$arg" == --port=* ]]; then
        # Extract port argument
        PORT_ARG="$arg"
    else
        # Add all other arguments
        ARGS="$ARGS $arg"
    fi
done

# Build command with all detected options
if [[ -n "$COMMAND" && "$COMMAND" == "serve" ]]; then
    # For 'serve' command, use server.sh
    FINAL_CMD="$INSTALL_DIR/server.sh"

    # Add port if specified
    if [[ -n "$PORT_ARG" ]]; then
        FINAL_CMD="$FINAL_CMD $PORT_ARG"
    fi

    # Add no-conda flag if specified
    if [[ "$USE_UV" == false ]]; then
        FINAL_CMD="$FINAL_CMD --no-uv"
    fi
    if [[ "$USE_CONDA" == false ]]; then
        FINAL_CMD="$FINAL_CMD --no-conda"
    fi

    # Add any remaining args
    FINAL_CMD="$FINAL_CMD $ARGS"
else
    # For all other commands, use client.sh
    FINAL_CMD="$INSTALL_DIR/client.sh"

    # Add port if specified
    if [[ -n "$PORT_ARG" ]]; then
        FINAL_CMD="$FINAL_CMD $PORT_ARG"
    fi

    # Add no-conda flag if specified
    if [ "$USE_UV" == false ]; then
        FINAL_CMD="$FINAL_CMD --no-uv"
    fi
    if [ "$USE_CONDA" == false ]; then
        FINAL_CMD="$FINAL_CMD --no-conda"
    fi

    # Add all other arguments
    FINAL_CMD="$FINAL_CMD $ARGS"
fi

# Execute the final command
eval $FINAL_CMD
EOF

sudo sed -i "s|{{USE_UV}}|$USE_UV|" /usr/local/bin/rkllama
sudo sed -i "s|{{USE_CONDA}}|$USE_CONDA|" /usr/local/bin/rkllama

sudo chmod +x /usr/local/bin/rkllama
echo -e "${CYAN}Executable created successfully: /usr/local/bin/rkllama${RESET}"

# Display statuses and available commands
echo -e "${GREEN}+ Configuration: OK.${RESET}"
echo -e "${GREEN}+ Installation : OK.${RESET}"

echo -e "${BLUE}Server${GREEN}  : $INSTALL_DIR/server.sh $CONDA_ARG${RESET}"
echo -e "${BLUE}Client${GREEN}  : $INSTALL_DIR/client.sh $CONDA_ARG${RESET}\n"
echo -e "${BLUE}Global command  : ${RESET}rkllama"
