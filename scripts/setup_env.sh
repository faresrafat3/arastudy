#!/bin/bash
echo "🔧 Setting up Bug Hunter environment..."

# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 5. Install project dependencies
pip install -r requirements.txt

# 6. Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "✅ Setup complete!"
