# AraStudy Research Log

# ════════════════════════════════
# Phase 3: Fresh Start (March 2026)
# All previous results archived as pilot experiments.
# Starting systematic experiments from scratch.
# ════════════════════════════════

سجّل كل تجربة هنا. التوثيق هو أساس البحث العلمي!

## Template

### تجربة [رقم] — [التاريخ]

**الفرضية (Hypothesis):**
(إيه اللي بتختبره ولماذا؟)

**الإعدادات (Settings):**
- **الداتا:** (مثلاً: Wiki-only, 100M tokens)
- **المعمارية:** (مثلاً: 30M params, 6 layers)
- **Hyperparameters:** (LR: 1e-3, Batch: 64)
- **الهاردوير:** (RTX 4050)

**النتائج (Results):**
- **Perplexity:** 
- **OALL Score:** 
- **Training Loss:** (أرفق صورة أو رابط WandB)
- **ملاحظات:** (مشاكل، مفاجآت)

**الخطوة الجاية (Next Step):**
(بناءً على النتيجة، هتعمل إيه؟)

---

## سجل التجارب (Experiments Log)

(ابدأ التسجيل من الأسفل)

---

## Exp01 — Arabic Tokenization Impact on Ultra-Small LM

### الفرضية الأساسية
أفضل tokenizer عربي لموديل 30M غالباً يكون `Morph-BPE` مع vocab متوسطة (16K) لأنه يوازن بين ضغط المفردات وكفاءة التمثيل.

### قائمة الـ Runs المطلوبة
- `BPE-32K`
- `BPE-16K`
- `BPE-8K`
- `Morph-BPE-16K`
- `Morph-BPE-8K`
- `Char-Level`

### ثوابت لازم لا تتغير بين الـ runs
- نفس البيانات وتقسيماتها.
- نفس المعمارية الأساسية.
- نفس seed.
- نفس optimizer/scheduler/training budget.

### Template سريع لكل Run
**Run Name:**
**Date:**
**Tokenizer:**
**Config Hash/Path:**
**Seed:**
**GPU:**

**Tokenizer Stats (قبل التدريب):**
- Vocab size:
- Avg Tok/Word:
- Avg Tok/Sent:
- Tokens/Char:
- Embedding Params:

**Training Results (بعد التدريب):**
- Best Val PPL:
- Tokens/sec:
- Peak VRAM:
- Best Step:

**Fair Comparison Metric:**
- BPC = log2(PPL) × tokens_per_char =

**Observations:**
-

**Next Action:**
-

---

## Phase 1: Tokenizer Analysis — 2026-03-16

### Completed:
- Data: 12,000 raw → 4,710 clean lines (wiki_ar_sample)
- 6 tokenizers trained successfully
- Analysis report generated

### Key Finding:
- BPE-32K: Tok/Word=1.2989, Tok/Sent=87.7288, Embed Ratio=0.394259
- BPE-16K: Tok/Word=1.4046, Tok/Sent=94.8686, Embed Ratio=0.245531
- BPE-8K: Tok/Word=1.5622, Tok/Sent=105.5127, Embed Ratio=0.139946
- Morph-BPE-16K: Tok/Word=1.6812, Tok/Sent=113.5508, Embed Ratio=0.245531
- Morph-BPE-8K: Tok/Word=1.8361, Tok/Sent=124.0127, Embed Ratio=0.139946
- Char-Level: Tok/Word=6.0274, Tok/Sent=407.1017, Embed Ratio=0.001503

### Observation:
- Dataset too small (4,710 lines) for final conclusions
- Tokenizer stats may shift with larger data
- BPE-32K embedding ratio is high for a 30M model budget

### Next Steps:
- Option A: Sanity check with mini-models (5-10M)
- Option B: Collect larger dataset and retrain tokenizers
- Priority: verify full training/eval pipeline end-to-end on cheap runs first

---

## Phase 1 Analysis — Predictions (2026-03-16)

### Surprising finding (to validate in training)
Morph-BPE produced **more** tokens per word than plain BPE at the same vocab size:
- 16K: Morph-BPE-16K = 1.6812 vs BPE-16K = 1.4046 (**+19.7%**)
- 8K: Morph-BPE-8K = 1.8361 vs BPE-8K = 1.5622 (**+17.5%**)

Interpretation: segmentation increases sequence length, but may still improve learning if morpheme-level units encode reusable grammatical patterns.

### Trade-off snapshot (for 30M design intuition)
- BPE-32K: shortest sequences, but highest embedding cost (39.4%).
- BPE-16K: balanced setting (moderate sequence length, moderate embedding cost).
- BPE-8K: strong capacity candidate (low embedding cost, higher transformer share).
- Morph-BPE-16K: same embedding share as BPE-16K, but longer sequences.
- Morph-BPE-8K: same embedding share as BPE-8K, but longest sequences among BPE variants.
- Char-Level: minimal embedding cost, but sequence length likely too long for useful context coverage.

### Effective capacity view (context=512)
| Tokenizer | Tok/Sent | Sentences in 512 | Transformer% |
|---|---:|---:|---:|
| BPE-32K | 87.7288 | 5.84 | 60.57% |
| BPE-16K | 94.8686 | 5.40 | 75.45% |
| BPE-8K | 105.5127 | 4.85 | 86.01% |
| Morph-BPE-16K | 113.5508 | 4.51 | 75.45% |
| Morph-BPE-8K | 124.0127 | 4.13 | 86.01% |
| Char-Level | 407.1017 | 1.26 | 99.85% |

Where:
- `Sentences in 512 = 512 / Tok-Sent`
- `Transformer% = 1 - EmbeddingRatio`

### Predictions (best → worst, to be tested)
1. BPE-8K
2. Morph-BPE-8K or BPE-16K
3. Morph-BPE-16K
4. BPE-32K
5. Char-Level

### Testable hypotheses for Phase 2
- H1: Embedding ratio matters more than Tok/Word in 30M models.
- H2: Morph segmentation may hurt sequence efficiency but improve quality (BPC).
- H3: 8K vocab is the strongest efficiency-quality sweet spot for 30M Arabic models.
- H4: BPE-32K underperforms despite best Tok/Word due to parameter budget pressure.

### Data-size caveat (critical)
Current corpus (4,710 clean lines) is sufficient for pipeline/sanity signals, not for publishable conclusions.
All rankings above are preliminary until tokenizer retraining and model training on larger data.

---

## Phase 2A Setup — Sanity Check Training (2026-03-16)

### الهدف
تشغيل تدريب مصغر (5-10M تقريباً) لكل tokenizer للتأكد إن الـ pipeline كامل شغال قبل التدريب الكبير.

### ما تم تجهيزه
- Config: `configs/experiments/exp01_sanity_train.yaml`
- Single run trainer: `src/training/train_exp01_sanity.py`
- All-tokenizers runner: `scripts/run_exp01_sanity_all.sh`
- Leaderboard aggregator: `src/training/summarize_exp01_sanity.py`

### إعدادات sanity الحالية
- Model: dim=256, layers=4, heads=4, context=128
- Training: 300 steps, batch=8, grad_accum=2, lr=3e-4
- Evaluation: every 50 steps

### Smoke Run (completed)
- Tokenizer: BPE-16K
- Best Val Loss: 8.006859
- Params: 7,506,176
- Tokens Seen: 614,400
- Time: 7.24s

### مخرجات المرحلة
- `results/logs/exp01_sanity/bpe_16k_summary.txt`
- `results/logs/exp01_sanity/bpe_16k_metrics.csv`
- `results/logs/exp01_sanity/sanity_leaderboard.md`

### الخطوة التالية المباشرة
شغّل كل الـ 6 tokenizers بنفس الإعداد:
`bash scripts/run_exp01_sanity_all.sh`

### Phase 2A Sanity Results (all 6 runs)
Leaderboard by best validation loss (sanity only):
1. Char-Level: 2.519621
2. Morph-BPE-8K: 6.700375
3. Morph-BPE-16K: 7.040647
4. BPE-8K: 7.591602
5. BPE-16K: 8.006859
6. BPE-32K: 8.331104

Saved at:
- `results/logs/exp01_sanity/sanity_leaderboard.md`

Method note:
- Raw CE loss across different tokenizers is not a final fair comparison.
- Final comparison must use tokenizer-normalized metrics (e.g., BPC).
- Sanity objective here is pipeline validation + preliminary ranking signal only.

---

## Phase 2A — Full Analysis (2026-03-16)

### Generation Quality Analysis
- BPE-32K / BPE-16K: outputs contain more recognizable words but still high repetition and weak coherence.
- BPE-8K: more visible subword fragmentation with small-data instability.
- Morph-BPE-16K / Morph-BPE-8K: lower val loss but morpheme-fragment style generations, weak composition at this model size.
- Char-Level: lowest val loss in sanity setting but poor textual readability.

### UNK Issue (fixed in sanity generation)
- Root cause: prompt text was not normalized like training corpus, while tokenizer learned normalized Arabic forms.
- Fix applied: generation now normalizes prompts using corpus normalization config and can disallow sampling `unk` token.
- Validation: unknown marker count in regenerated samples = 0.
- Output file: `results/logs/exp01_sanity/generation_samples.md`.

### Key Discovery #1 — Morph Loss vs Generation Paradox
- Morph-BPE improves sanity val loss versus same-vocab BPE.
- Yet word-level readability is weaker in generation at this small model/data budget.
- Working hypothesis: model learns morpheme prediction before mastering composition.

### Key Discovery #2 — Embedding Ratio Pressure
- BPE-32K has best Tok/Word but worst sanity val loss among BPE settings.
- High embedding share appears costly for tiny-model capacity.

### Key Discovery #3 — Morph Speed Overhead
- At same vocab size, Morph-BPE and BPE run at near-identical wall time in sanity training.
- Practical implication: morphology preprocessing is not adding noticeable training-time overhead here.

### Fairness Check
- Same seed, steps, LR schedule, warmup, batch, grad accumulation, optimizer, and model backbone across runs.
- Only tokenizer choice changes (thus vocab and embedding size change by design).

### Phase 2B Priority List
1. Collect larger corpus (target 50M+ tokens cleaned).
2. Retrain all tokenizers on larger corpus.
3. Add tokenizer-normalized evaluation (BPC) in model evaluation.
4. Run full 30M training with fixed compute budget across tokenizers.
5. Perform structured generation-quality assessment (fluency/coherence/grammar).

---

## Phase 2B/2C Hybrid Readiness Update (2026-03-16)

### Implemented Safeguards
- Streaming Wikipedia ingestion with max-article cap and disk preflight checks.
- Chunked cleaning with SQLite disk-based dedup and resume-friendly processed-file tracking.
- SentencePiece large-corpus knobs enabled (`input_sentence_size`, shuffle, large-corpus mode).
- Full training now supports checkpoint resume (`--resume`) for Colab interruptions.

### Full-Train Instrumentation Added
- BPC logging during evaluation.
- Tokens/second logging.
- Peak VRAM logging.
- Periodic generation samples (every N steps).
- Early stopping patience logic.

### New Execution Artifacts
- `configs/experiments/exp01_phase2b_data.yaml`
- `configs/experiments/exp01_full_train.yaml`
- `scripts/run_phase2b_data.sh`
- `scripts/run_phase2b_tokenizers.sh`
- `scripts/run_colab.sh`
- `scripts/run_phase2c_hybrid_schedule.sh`
- `notebooks/colab_train.ipynb`

---

## Phase 2B Complete — 2026-03-16

### Corpus Stats
- Lines: 1,390,451 (≈295.2x larger than Phase 1: 4,710)
- Words: 84,025,327
- Characters: 488,667,555
- Estimated tokens: ~100M–120M
- Split: 1,251,640 train / 69,439 valid / 69,372 test

### Tokenizer Stats (Phase 2B)
| Tokenizer | Vocab | Tok/Word | Delta vs Phase 1 |
|---|---:|---:|---:|
| BPE-32K | 32000 | 1.348 | +0.049 |
| BPE-16K | 16000 | 1.480 | +0.075 |
| BPE-8K | 8000 | 1.655 | +0.093 |
| Morph-BPE-16K | 16000 | 1.764 | +0.082 |
| Morph-BPE-8K | 8000 | 1.919 | +0.083 |
| Char-Level | 188 | 6.063 | +0.036 |

### Confirmed Findings
- H6 confirmed: small-corpus tokenizer stats were misleading.
- All tokenizers increased in Tok/Word on large data (expected OOV/diversity effect).
- Relative ranking remained stable: BPE-32K < BPE-16K < BPE-8K < Morph-BPE-16K < Morph-BPE-8K < Char.
- Tok/Sent decreased across all tokenizers, indicating shorter average sentence units in Phase 2B text mix.

### Embedding Ratio Snapshot (Current PE-aware model)
Model base (shared): core=20,453,888 params, positional embedding=262,144 params (`dim=512`, `layers=6`, `heads=8`, `max_seq_len=512`).

| Tokenizer | Vocab | Embedding Params | Total Params | Embedding % |
|---|---:|---:|---:|---:|
| BPE-32K | 32000 | 16,384,000 | 37,100,032 | 44.16% |
| BPE-16K | 16000 | 8,192,000 | 28,908,032 | 28.34% |
| BPE-8K | 8000 | 4,096,000 | 24,812,032 | 16.51% |
| Morph-BPE-16K | 16000 | 8,192,000 | 28,908,032 | 28.34% |
| Morph-BPE-8K | 8000 | 4,096,000 | 24,812,032 | 16.51% |
| Char-Level | 188 | 96,256 | 20,812,288 | 0.46% |

### Decision
GO for Phase 2C full training.

Readiness checks passed:
- Phase 2B split paths wired in full-train config.
- Phase 2B tokenizers wired in full-train config.
- Learned positional embedding integrated in model.
- Resume checkpoints enabled (`--resume`).
- BPC logged during evaluation (validation split).
- Generation prompts externalized in `generation_benchmark_prompts.md`.

### Phase 2C Schedule (Hybrid)
- Day 1: local `morph_bpe_8k` + Colab `bpe_8k`
- Day 2: local `bpe_16k` + Colab `morph_bpe_16k`
- Day 3: local `bpe_32k` + Colab `char`

---

## Phase 2C — Partial Results (2026-03-17)

### Completed Runs (3/6)
| Tokenizer | Best Val Loss | Best BPC (corrected) | Best Step | Final Step | Params | Avg Tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Morph-BPE-8K | 3.322887 | 1.829851 | 13000 | 14999 | 24.8M | 52,181 | 2.64 GB |
| BPE-16K | 4.072959 | 1.729788 | 16500 | 18204 | 28.9M | 46,470 | 3.55 GB |
| BPE-32K | 4.403984 | 1.703897 | 12500 | 14499 | 37.1M | 39,267 | 2.83 GB |

### Major Finding
- Corrected BPC ranking reverses raw val-loss ranking on current partial set (3/6).
- Raw val loss alone would lead to a different model choice.
- For current evidence, BPE-32K shows best BPC despite highest embedding share and slower throughput.

### Generation (Preliminary)
- All three completed runs generate coherent Arabic paragraphs with clear improvement over Phase 2A.
- BPE-16K appears strongest in qualitative coherence; BPE-32K close but with more repetition; Morph-BPE-8K shows more topic drift.
- Formal human evaluation is still required before final claims.

### Pending Runs (Critical)
- BPE-8K (highest priority; fair pair for Morph-BPE-8K)
- Morph-BPE-16K (pair with BPE-16K)
- Char-Level (lower bound)


