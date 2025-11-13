#!/usr/bin/env bash
set -euo pipefail

# Location of this script (project root: NeuralNetworksProject)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the venv-creation script
CREATE_VENV_SCRIPT="$SCRIPT_DIR/Codebase/Setup/create_venv.sh"

echo "Project root:          $SCRIPT_DIR"
echo "create_venv.sh path:   $CREATE_VENV_SCRIPT"
echo

# Make sure the script exists
if [ ! -f "$CREATE_VENV_SCRIPT" ]; then
    echo "ERROR: create_venv.sh not found at:"
    echo "  $CREATE_VENV_SCRIPT"
    exit 1
fi

# Ensure it's executable
chmod +x "$CREATE_VENV_SCRIPT"

# Run it
"$CREATE_VENV_SCRIPT"

echo
echo "Setup complete."
echo "To activate the virtual environment, run:"
echo "  source \"$SCRIPT_DIR/.venv/bin/activate\""
