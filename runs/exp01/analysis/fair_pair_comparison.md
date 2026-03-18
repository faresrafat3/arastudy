# Fair Pair Comparison (Exp01)

## 8K Pair (same params: 24.8M)

- BPE-8K val loss: 3.841
- Morph-BPE-8K val loss: 3.323
- Relative improvement (Morph): ~13.5%

- BPE-8K corrected BPC: 1.825
- Morph-BPE-8K corrected BPC: 1.830
- Relative difference: ~0.27% (near tie)

Interpretation:
Morphological segmentation improves token-level prediction at 8K, but BPC remains effectively tied due to token-count normalization.

## 16K Pair (same params: 28.9M)

- BPE-16K val loss: 4.073
- Morph-BPE-16K val loss: 3.476
- Relative improvement (Morph): ~14.7%

- BPE-16K corrected BPC: 1.730
- Morph-BPE-16K corrected BPC: 1.760
- Relative difference: ~1.7%

Interpretation:
At larger vocabulary, morphology still improves token-level loss, but BPE remains better in character-level compression.
