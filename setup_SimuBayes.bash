#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Print the current directory
echo "Current directory: $CURRENT_DIR"

# Check if requirements.txt exists
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$CURRENT_DIR/requirements.txt"
else
    echo "requirements.txt not found in the current directory."
    exit 1
fi

# Check if SimuBayes.sh exists
if [ -f "$CURRENT_DIR/SimuBayes.sh" ]; then
    find . -name "SimuBayes.sh" -exec chmod +x {} \;
    echo "Setting up alias for SimuBayes.sh..."
    # Create an alias for SimuBayes.sh
    alias SimuBayes="$CURRENT_DIR/SimuBayes.sh"
    echo "Alias 'SimuBayes' created. You can now use 'SimuBayes' to start the app."
else
    echo "SimuBayes.sh not found in the current directory."
    exit 1
fi

# Add the alias to ~/.bashrc for persistence
echo "Adding alias to ~/.bashrc for future sessions..."
echo "alias simubayes='$CURRENT_DIR/SimuBayes.sh'" >> ~/.bashrc

# Reload ~/.bashrc to apply the alias immediately
source ~/.bashrc

echo "Setup complete! You can now use the 'SimuBayes' command to start the app."
