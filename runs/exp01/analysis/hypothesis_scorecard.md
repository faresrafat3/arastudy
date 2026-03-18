# Exp01 Hypothesis Scorecard

- H1: Morph improves BPC over BPE at same vocab
  - Status: Nuanced
  - Evidence: Near tie at 8K; rejected at 16K.

- H2: Smaller vocab is better for tiny models
  - Status: Rejected
  - Evidence: BPC ranking follows larger vocab in this setup.

- H3: Morph loss better but generation worse
  - Status: Nuanced
  - Evidence: Mixed by vocab size; improved val_loss is consistent.

- H4: Char-level underperforms under tiny budget
  - Status: Confirmed
  - Evidence: Worst BPC with weak generation quality.

- H5: Positional encoding critically affects generation
  - Status: Confirmed
  - Evidence: Strong generation jump after adding learned positional embeddings.

- H6: Small-corpus tokenizer stats can mislead
  - Status: Confirmed
  - Evidence: Stats shifted after scaling corpus in Phase 2B.

- H7: BPC is essential for fair tokenizer comparison
  - Status: Confirmed
  - Evidence: Val-loss and BPC induce different rankings.
