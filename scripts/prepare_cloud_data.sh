#!/usr/bin/env bash
set -euo pipefail

echo "Preparing data for cloud upload..."

rm -rf cloud_data
mkdir -p cloud_data/splits/phase2b
mkdir -p cloud_data/tokenizers/phase2b
mkdir -p artifacts/datasets

for f in data/splits/phase2b/train.txt data/splits/phase2b/valid.txt data/splits/phase2b/test.txt; do
  if [[ ! -f "$f" ]]; then
    echo "❌ Missing required split file: $f"
    exit 1
  fi
done

cp data/splits/phase2b/train.txt cloud_data/splits/phase2b/
cp data/splits/phase2b/valid.txt cloud_data/splits/phase2b/
cp data/splits/phase2b/test.txt cloud_data/splits/phase2b/

for tok in bpe_16k bpe_32k morph_bpe_8k bpe_8k morph_bpe_16k char; do
  if [[ ! -f "results/tokenizers/phase2b/${tok}.model" ]] || [[ ! -f "results/tokenizers/phase2b/${tok}.vocab" ]]; then
    echo "❌ Missing tokenizer files for: ${tok}"
    exit 1
  fi
  cp "results/tokenizers/phase2b/${tok}.model" cloud_data/tokenizers/phase2b/
  cp "results/tokenizers/phase2b/${tok}.vocab" cloud_data/tokenizers/phase2b/
done

zip -r arastudy_cloud_data.zip cloud_data/ >/dev/null

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('.')
zip_path = root / 'arastudy_cloud_data.zip'
cloud_root = root / 'cloud_data'

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

files = []
for p in sorted(cloud_root.rglob('*')):
    if p.is_file():
        files.append(
            {
                'path': str(p.relative_to(root)),
                'size_bytes': p.stat().st_size,
                'sha256': sha256(p),
            }
        )

manifest = {
    'dataset_name': 'arastudy_phase2b_cloud_bundle',
    'version': 'phase3_v1',
    'description': 'Phase 2B splits + 6 tokenizers for cloud training',
    'zip_file': str(zip_path),
    'zip_size_bytes': zip_path.stat().st_size,
    'zip_sha256': sha256(zip_path),
    'targets': {
        'kaggle_dataset': 'arastudy-phase2b-data',
        'colab_drive_path': 'MyDrive/arastudy_data/',
        'lightning_workspace_path': '/teamspace/studios/this_studio/data/'
    },
    'files': files,
}

out = root / 'artifacts/datasets/arastudy_cloud_data_manifest.json'
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'wrote manifest: {out}')
PY

echo "✅ ZIP ready: $(du -sh arastudy_cloud_data.zip | awk '{print $1}')"
echo "✅ Manifest ready: artifacts/datasets/arastudy_cloud_data_manifest.json"
