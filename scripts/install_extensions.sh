#!/bin/bash
echo "📦 Installing VS Code Extensions..."

# === Core AI Extensions ===
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension Continue.continue

# === Python & ML ===
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.vscode-ai

# === Code Quality ===
code --install-extension charliermarsh.ruff
code --install-extension ms-python.mypy-type-checker
code --install-extension njpwerner.autodocstring

# === Git & Collaboration ===
code --install-extension eamodio.gitlens
code --install-extension GitHub.vscode-pull-request-github
code --install-extension mhutchie.git-graph

# === Productivity ===
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension gruntfuggly.todo-tree
code --install-extension aaron-bond.better-comments
code --install-extension mechatroner.rainbow-csv

# === Remote & Containers ===
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-wsl
code --install-extension ms-azuretools.vscode-docker

# === Visualization ===
code --install-extension RandomFractalsInc.vscode-data-preview
code --install-extension tomoki1207.pdf

echo "✅ Extensions installation commands executed!"
