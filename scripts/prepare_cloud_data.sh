#!/usr/bin/env bash
set -euo pipefail

echo "Preparing data for cloud upload..."

mkdir -p cloud_data/splits/phase2b
mkdir -p cloud_data/tokenizers/phase2b

cp data/splits/phase2b/train.txt cloud_data/splits/phase2b/
cp data/splits/phase2b/valid.txt cloud_data/splits/phase2b/
cp data/splits/phase2b/test.txt cloud_data/splits/phase2b/

for tok in bpe_16k bpe_32k morph_bpe_8k bpe_8k morph_bpe_16k char; do
  cp "results/tokenizers/phase2b/${tok}.model" cloud_data/tokenizers/phase2b/
  cp "results/tokenizers/phase2b/${tok}.vocab" cloud_data/tokenizers/phase2b/
done

zip -r arastudy_cloud_data.zip cloud_data/ >/dev/null

echo "✅ ZIP ready: $(du -sh arastudy_cloud_data.zip | awk '{print $1}')"
