# Datasets for Cloud Services (Phase 3)

This file defines the exact data bundle to upload to cloud services for AraStudy Phase 3.

## Bundle Name
- `arastudy_cloud_data.zip`

## Includes
- `cloud_data/splits/phase2b/train.txt`
- `cloud_data/splits/phase2b/valid.txt`
- `cloud_data/splits/phase2b/test.txt`
- `cloud_data/tokenizers/phase2b/bpe_16k.{model,vocab}`
- `cloud_data/tokenizers/phase2b/bpe_32k.{model,vocab}`
- `cloud_data/tokenizers/phase2b/bpe_8k.{model,vocab}`
- `cloud_data/tokenizers/phase2b/morph_bpe_16k.{model,vocab}`
- `cloud_data/tokenizers/phase2b/morph_bpe_8k.{model,vocab}`
- `cloud_data/tokenizers/phase2b/char.{model,vocab}`

## How to build
- Run: `./scripts/prepare_cloud_data.sh`

## Generated manifest
- `artifacts/datasets/arastudy_cloud_data_manifest.json`
- Contains file list, size, and SHA-256 for each file + zip checksum.

## Upload targets
- Kaggle Dataset: `arastudy-phase2b-data`
- Colab (Drive): `MyDrive/arastudy_data/`
- Lightning AI: `/teamspace/studios/this_studio/data/`

## Notes
- Do not regenerate splits/tokenizers for this phase.
- This bundle is the canonical upload payload for Exp01 cloud runs.
