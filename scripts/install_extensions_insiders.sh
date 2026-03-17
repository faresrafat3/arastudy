#!/bin/bash
echo "📦 Installing VS Code Insiders Extensions..."

# Check if code-insiders is available
if ! command -v code-insiders &> /dev/null; then
    echo "❌ Error: 'code-insiders' command not found."
    echo "   Please make sure VS Code Insiders is installed and added to your PATH."
    exit 1
fi

# === Core AI Extensions ===
code-insiders --install-extension GitHub.copilot
code-insiders --install-extension GitHub.copilot-chat
code-insiders --install-extension Continue.continue

# === Python & ML ===
code-insiders --install-extension ms-python.python
code-insiders --install-extension ms-python.vscode-pylance
code-insiders --install-extension ms-toolsai.jupyter
code-insiders --install-extension ms-toolsai.vscode-ai

# === Code Quality ===
code-insiders --install-extension charliermarsh.ruff
code-insiders --install-extension ms-python.mypy-type-checker
code-insiders --install-extension njpwerner.autodocstring

# === Git & Collaboration ===
code-insiders --install-extension eamodio.gitlens
code-insiders --install-extension GitHub.vscode-pull-request-github
code-insiders --install-extension mhutchie.git-graph

# === Productivity ===
code-insiders --install-extension streetsidesoftware.code-spell-checker
code-insiders --install-extension gruntfuggly.todo-tree
code-insiders --install-extension aaron-bond.better-comments
code-insiders --install-extension mechatroner.rainbow-csv

# === Remote & Containers ===
code-insiders --install-extension ms-vscode-remote.remote-ssh
code-insiders --install-extension ms-vscode-remote.remote-wsl
code-insiders --install-extension ms-azuretools.vscode-docker

# === Visualization ===
code-insiders --install-extension RandomFractalsInc.vscode-data-preview
code-insiders --install-extension tomoki1207.pdf

echo "✅ All extensions installed for VS Code Insiders!"
