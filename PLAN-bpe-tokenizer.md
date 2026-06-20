# PLAN — BPE Tokenizer for NER (HW2)

## Context
Grounded in `HW2_NLP_Translation.md` + `code_env_example/README.md`, inspired by the lecture
(slides 31-45: Sennrich/HF word-pretokenized char-BPE) and the tutorial (Karpathy byte-level BPE).
We implement `code/bpe_tokenizer.py` (the only graded code), train 3 tokenizers, and a fixed bi-LSTM
NER model runs on top. A first char-level attempt scored **F1 0.4287 on domain_1 dev — below the
0.5 gate** — so this plan centers on an empirical bake-off to find a design that clears 0.5 and wins
the efficiency/speed/F1 competition.

Decision: **don't pre-commit to one method — implement both and measure.**

## Status & Results (updated 2026-06-20)
Bake-off complete (vocab 5000, full data unless noted, fixed bi-LSTM NER, best-of-20-epoch F1):
 
| Config | tokens/char | speed (tok/s) | F1 | Gate |
| :-- | :-- | :-- | :-- | :-- |
| domain_1 · word · ws · nb5 | **0.2876** | 205k | **0.4481** | ❌ <0.5 |
| domain_1 · word · regex · nb5 | 0.3128 | 262k | 0.4163 | ❌ regex hurts |
| domain_1 · byte (2 MB sample) | 0.3210 | 19k | 0.4135 | ❌ byte loses all 4 |
| domain_1 · word · ws · nb5 · **lowercase** | 0.2707 | — | 0.3340 | ❌ case is a key cue |
| domain_2 · word · regex · nb5 | 0.2976 | 238k | **0.9605** | ✅ |

**Decisions locked**: `method="word"`, `pretok="ws"`, `num_bigrams=5` (= current `BPETokenizer`
defaults). Byte method and regex pre-tok both rejected (lost on F1 + efficiency).
**Engineering done**: `_train_word` rewritten with **incremental pair-counts** (inverted index) —
full domain_1 trains in ~700 s (was ~22 min naive). Encode is per-word cached.

**Standing**: tokenizer_2 (domain_2) clears the gate at 0.96. **tokenizer_1 (domain_1 / Twitter) is
stuck at 0.448 < 0.5** — the one open blocker. tokenizer_3 (hidden domain) not built yet.
**VM budget exhausted** (`time_left` went negative, machine died mid-run twice). The cased@10k F1
diagnostic was launched but the machine died before it logged a result — re-fetch `~/HW2/v10.log`
first thing next session (home dir persists; `/tmp` does not).

**Vocab size IS allowed to change** (grep of `HW2_NLP_Translation.md`: the only "token limit" is the
≤bigram *span* rule; no vocab cap; 5000 is just generate_tokenizers' CLI default). Mechanism added:
`FORCE_VOCAB_SIZE` constant in `bpe_tokenizer.py` (default `None` = honor caller, so experiments still
sweep vocab; set to e.g. 10000 to force our operating vocab so the submission reproduces even if the
grader re-runs generate_tokenizers with defaults). **Flip only after F1-validating the larger vocab**
(cased@10k F1 still TBD).

**Why domain_1 is hard**: Twitter's huge type count (typos, `@handles`, `#tags`, `loooove`) eats the
5000-merge budget → entity words fragment → only a generic first-subtoken carries the label →
low recall. regex/bigram tweaks don't address this root cause.

## Open questions (resolve with staff / PDF)
1. **Vocab policy** — *partly resolved*: changing vocab is **allowed** (no rule). `FORCE_VOCAB_SIZE`
   makes any vocab reproduce via generate_tokenizers. Remaining nicety: confirm grader either uses our
   submitted `.pkl` or re-runs generate_tokenizers (FORCE covers both). Then F1-validate cased@10k.
2. **Real F1 thresholds**: the "0.5 / 1 on domain_2" wording is likely mistranslated — confirm the
   actual domain_1 bar (we may already pass at 0.448).

## Hard constraints (verified against source + instructions)
1. **Char-start BPE** ("break sentence to character level"), merge upward.
2. **Tokens capped at the bigram level** — at most **two adjacent words** per token (≤1 internal
   space). A 3-word token is **disqualifying**. Must also produce **≥1 bigram** per tokenizer
   (`check_submission.py:263-279`).
3. **`space_token`** set (non-None) and present in `token_to_id` (NER `train_ner_model.py:49`,
   checker `:254`). e.g. `'_'`, `'▁'`, or `' '`.
4. **Reproducible via `generate_tokenizers.py`**, which calls `Tokenizer(vocab_size=5000)`. Treat
   vocab 5000 as the operating point **unless the vocab policy (open Q#1) says submitted pkls are used**.
   Training must finish within the 10h machine budget (`time_left` to check).
5. **Pickleable**; loadable by scripts that only know `BaseTokenizer`.
6. **decode** must round-trip characters and map the space marker back to `' '` — the NER pipeline
   aligns tokens to words via `decode([id])` / `decode(ids[:i])` (`train_ner_model.py:117-154`).
7. Libs: `numpy/regex/torch/tqdm` only; provided data only.
8. **Editable & submittable**: `bpe_tokenizer.py` (+ helpers), and `train_tokenizer.py`,
   `test_tokenizer.py`. **Never edit/submit**: `generate_tokenizers.py`, `train_ner_model.py`,
   `base_tokenizer.py`, `check_submission.py`.
9. Data on disk: `data/data/domain_{1,2}_{train,dev}.txt`,
   `data/data/ner_data/{train,dev}_{1,2}_binary.tagged` (instructions' `tagged.train_1_binary`
   naming is reversed — use the real on-disk names).

Grade: Impl 20% (F1≥0.5 domain_1 dev; domain_2 similarly — wording ambiguous, treat as ≥0.5) ·
Competition 45% (efficiency + speed + hidden-domain F1, equal weight, tradeoffs) ·
Tokenizer-3 eval 15% (report) · Report 20% (strict format).

## Design — one class, two selectable methods
`BPETokenizer(BaseTokenizer)` with `method` param (`"word"` | `"byte"`), so we A/B test fairly and
bake the winner in as the default (since `generate_tokenizers.py` only passes `vocab_size`).
Both methods satisfy all hard constraints above.

### Method A — lecture-style word-pretokenized char-BPE (primary, matches "as learned in lecture")
- Pre-tokenize lines into words; represent each word with a leading space marker (`'_word'`, HF style).
- Init vocab = all unique characters (+ marker). Count adjacent-pair freqs over a **unique-word
  frequency dict** (dedup → fast on 91 MB). Merge most-frequent pair, store ordered **rules+rank**,
  repeat to ~`vocab_size − (bigram budget)`.
- **Inference (slide 39)**: split to words; per word, init symbols, repeatedly apply the in-vocab
  merge with the best rank until none; **cache word→token-ids** (big speed win, Zipf).
- **Bigram phase (minimal)**: add the top few adjacent word-pairs (K≈1-5) as bigram tokens capped at
  ≤2 words. Keep K small — every bigram spends a vocab slot and merges two words into one token (the
  2nd word then can't carry its own first-subtoken label → hurts NER). Record top-5 / least-5 bigrams
  for the report.
- Unseen char at encode → `[UNK]` (rare on this data).

### Method B — tutorial byte-level full-stream BPE (alternative; robust, lossless)
- UTF-8 bytes (offset +4 past specials), full-stream merges, greedy encode by rank, `decode` via
  `bytes.decode("utf-8", errors="replace")` → **no [UNK]**, lossless.
- **Fix for this plan**: enforce the ≤2-word cap (skip any merge whose token would contain ≥2 spaces)
  and keep ≥1 bigram. Train on a deterministic sample (naive full-data is infeasible) + cache encode.

## Empirical bake-off (core of this plan) — all at vocab 5000
For each method, on domain_1 and domain_2:
1. Train via our trainer (Method A: full data; Method B: sample).
2. `test_tokenizer.py` → tokens/char (efficiency), encode speed, reconstruction%.
3. `train_ner_model.py` → **best dev F1**.
Compare in a table; **pick the method that clears F1≥0.5 on domain_1 and wins efficiency/speed**.
Lock it as the default `method`.

### Next steps for domain_1 F1 (0.448 → ≥0.5), ranked
Tried & rejected: regex pre-tok (hurts), byte method (loses), nb sweep (≈flat).
**Noise-normalization (@/URL/repeat-collapse) is OUT** — it changes char count → breaks the NER
length-based char-span alignment (cascade mislabel). Only **length-preserving** transforms are safe.

CPU-only entity-fragmentation proxy (8 MB sample; sub/word(ENTITY) = recall proxy, lower better):
| vocab | case | tok/char | sub/word(ENTITY) |
| :-- | :-- | :-- | :-- |
| 5000  | cased     | 0.2877 | 2.784 |
| 5000  | lowercase | 0.2709 | **2.426** |
| 10000 | cased     | 0.2615 | 2.513 |
| 10000 | lowercase | 0.2471 | **2.124** |

1. **Lowercase — REJECTED (GPU-tested 2026-06-20).** domain_1 lowercase@5000 F1 = **0.3340** vs cased
   **0.4481**. The proxy misled: fewer entity fragments, but losing capitalization as an entity cue
   dominated → worse recall (recon also drops to ~25%, expected). **Case is critical — keep it.**
2. **Vocab 10k, CASED** — the one remaining real lever; proxy cuts entity frag (2.784→2.513) WITHOUT
   touching case. Diagnostic d1 cased@10k F1 = **TBD**. Submittable only if vocab policy (open Q#1) allows.
3. **nb=1 full-data** — small, cheap confirm (nb sweep so far ≈flat).
4. **Confirm real threshold** (open Q#2) — may already pass at 0.448.

**Tried & rejected**: regex pre-tok (0.42), byte method (0.41), lowercase (0.33), noise-norm
(alignment-unsafe), nb sweep (≈flat). **domain_1 ceiling = 0.4481** (cased · word/ws/nb5 · vocab 5000).

### Done
- Bake-off (word vs byte, ws vs regex) — winner locked: word/ws/nb5.
- Incremental trainer (fast). Encode cache (speed). ≤2-word cap + ≥1 bigram enforced.
- Local dual-method sanity (round-trip, cap, pickle) green.

## Tokenizer 3 — hidden domain (15% + competition)
Train on domain_1 + domain_2, but **balance** them (domain_1 ≫ domain_2 in size → biases vocab):
downsample domain_1 / upweight domain_2. Prefer the more **robust** method (byte / byte-fallback)
since the hidden domain differs from both. Deterministic (seed 42 already set in generate_tokenizers).

## Competition optimizations (45%)
- **Efficiency** (tokens/sentence ↓): use the full merge budget well; larger effective subwords.
- **Speed** (encode ↑): per-word token cache; avoid O(n²) re-scans (precompute merge ranks).
- **F1**: per above. Note explicit tradeoffs in the report.

## Report (20%) — content to capture during experiments
Per-tokenizer training explanation + how it differs from base BPE + hidden-domain handling;
top-5/least-5 bigrams; dev F1 + efficiency per domain; tokenizer-3 hidden-domain evaluation.
Format: ≤1 A4 page, Arial 10, 2.54 cm margins, single column, 1.15 spacing, captioned centered images
(≥9 pt text). Format mismatch ⇒ grade 0 — follow exactly.

## Files
- **Edit**: `code/bpe_tokenizer.py` (both methods + helpers), optionally `code/train_tokenizer.py` /
  `code/test_tokenizer.py` (submittable). Keep dev-only scripts (`_sanity_bpe.py`, `_exp_train.py`,
  `_inspect.py`) **out of the submission zip**.
- **Never touch**: `generate_tokenizers.py`, `train_ner_model.py`, `base_tokenizer.py`,
  `check_submission.py`.

## Verification
1. **Local sanity** (both methods, tiny corpus, no GPU): `space_token` set + in vocab; ≥1 bigram and
   **no >2-word token** (assert ≤1 space in every token surface); `decode(encode(x))` round-trips;
   pickle save/load. Run with `PYTHONUTF8=1`.
2. **VM** (`~/HW2/code_env_example`, env built, data at `../data/data`):
   - `uv run python code/train_tokenizer.py --domain_file ../data/data/domain_1_train.txt --output_dir tokenizers --vocab_size 5000`
   - `uv run python code/test_tokenizer.py --tokenizer_path tokenizers/tokenizer.pkl --train_file ../data/data/domain_1_train.txt --test_file ../data/data/domain_1_dev.txt`
   - `uv run python code/train_ner_model.py --tokenizer_path tokenizers/tokenizer.pkl --train_file ../data/data/ner_data/train_1_binary.tagged --dev_file ../data/data/ner_data/dev_1_binary.tagged` → **F1≥0.5**
   - Repeat for domain_2. Bake-off table → pick method.
   - `uv run python generate_tokenizers.py --data_dir ../data/data` → all 3 `.pkl` at vocab 5000.
   - Build `HW2_<id>.zip` (code/ + trained_tokenizers/ + report) → `uv run python check_submission.py HW2_<id>.zip` → all `[OK]`/`[RESULT]`.
   - Use `tmux`; watch `time_left` (10h budget).
