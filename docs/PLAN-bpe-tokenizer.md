# PLAN — BPE Tokenizer for NER (HW2)

## What we built
`code/bpe_tokenizer.py` — one class `BPETokenizer(BaseTokenizer)`: lecture/HF-style word-pretokenized
char-BPE (`▁` space marker, char-init vocab, merge upward). **Training** is pure frequency-BPE;
**encoding** defaults to **greedy longest-match** over the trained vocab (trie, O(W)) — see below.

**Locked config**: `pretok="ws"`, `num_bigrams=5`, `encoder="longest"`, **`FORCE_VOCAB_SIZE=None`**
(honors the caller's `--vocab_size`). **Per-tokenizer vocab** (now allowed — Q&A): **domain_1=2000**
(F1, OOV-heavy Twitter), **domain_2=5000** + **domain_3=5000** (efficiency; their F1 is vocab-robust).
Reproduced via three `train_tokenizer.py` commands in `train_commands.txt` (NOT `generate_tokenizers.py`,
which forces one vocab for all 3) — staff explicitly permit this (Dvir, Jun 28). `train_tokenizer.py`
was extended to take multiple `--domain_file` (tok_3 = domain_1+domain_2) and an `--output_name`.

## How it works (current implementation)
`train` / `encode` / `decode` are the public methods (single word method — the old byte variant was removed).

### Train (`train`)
1. **Normalize**: collapse whitespace, space → `▁` marker. Length-preserving (keeps NER char-span alignment).
2. **Char-init vocab**: add `▁` + every unique char. *True BPE — start from chars, merge upward.*
3. **Unique-word freq dict**: dedup words; each `splits[word] = list(chars)`.
4. **Incremental merge loop**: keep `stats` (pair→count) + `index` (pair→words-containing-it, inverted
   index). Each step merge the most-frequent pair (tie-break by pair value), only re-touch affected words.
   Stop at `vocab_size − num_bigrams`. (Inverted index = the speed win: naive ~22 min → ~700 s full data.)
5. **Bigram phase** (`_add_word_bigrams`): count adjacent *word-pairs*, add top `num_bigrams` as single
   tokens — but only if both words are already 1 token each (so the merge fires). **Fallback force-adds
   the top pair if zero added → guarantees ≥1 bigram (else submission disqualified).** Also records
   top-5/least-5 bigrams for the report.

### Encode (`encode`)
Per word, pick tokens via the active `encoder`, **cached per word** (Zipf → big win):
- **`"longest"` (default)**: greedy maximal-munch over a prefix trie of the vocab (`_longest_match`,
  `_get_trie`) — O(W), ~+55–64% faster than merge-replay, slightly fewer tokens. Inference-only change
  (training stays pure BPE) → BPE-legal.
- **`"merge"`**: classic BPE replay (`_bpe_word`) — repeatedly apply the best-rank in-vocab merge.
Then `_apply_word_bigrams` runs a cross-word pass capped at ≤2 words. Unseen char → `[UNK]`.

### Decode (`decode`)
`[UNK]` → single `�` (length-preserving: 1 unseen char = 1 token = 1 char → no NER-alignment cascade).
Map `▁` back → real space. Round-trips.

## End-to-end example
Corpus: `"New York"` ×2, `"New Year"` ×1.

1. **Normalize** → `New▁York`, `New▁York`, `New▁Year`.
2. **Char vocab** → `{▁, N, Y, a, e, k, o, r, w}`.
3. **Splits**: `New→[N,e,w]`, `▁York→[▁,Y,o,r,k]`, `▁Year→[▁,Y,e,a,r]`.
4. **Merges** (freq-weighted): `(▁,Y)`=3 → `▁Y`; `(e,w)`=3 → `ew`; `(N,ew)`=3 → `New` (now `New`=1 token); …
5. **Bigram**: top word-pair `("New","▁York")`×2. `▁York` isn't 1 token yet → skip → fallback force-adds
   token `New▁York` (1 vocab token spanning 2 words). `_is_bigram_surface` → strip `▁`→space → 1 space ⇒ ≤2 words ✓.
6. **Encode `"New York"`** → chunks `[New, ▁York]` → `[New, ▁Yo, r, k]` → ids. (If `▁York` were 1 token,
   bigram `New▁York` fires → single id.)
7. **Decode** → `New▁York` → `New York`. ✓

**NER tie-in**: word `York`→`[▁Yo,r,k]`; *first-subtoken labeling* labels only `▁Yo`, rest get `-100`.
`space_token=▁` marks word boundaries. Entity fragmentation = root cause of the domain_1 F1 ceiling.

## How we got here (short)
A first plain char-level attempt scored F1 **0.4287** on domain_1 (below the original 0.5 gate), so we ran
an empirical bake-off (vocab 5000, fixed bi-LSTM, best-of-20-epoch F1) over pre-tokenization + casing:

| Config | tok/char | F1 | Verdict |
| :-- | :-- | :-- | :-- |
| domain_1 · ws · nb5 | 0.288 | 0.4481 | ✅ winner |
| domain_1 · regex | 0.313 | 0.4163 | ❌ regex hurts |
| domain_1 · **lowercase** | 0.271 | 0.3340 | ❌ case is a key cue |
| domain_2 · regex | 0.298 | 0.9605 | ✅ |

Then a **vocab sweep** (domain_1, cased word/ws/nb5) found a non-monotonic peak (early-code numbers;
superseded by the full new-code sweep in **Optimization** below — same conclusion, vocab 2000):

| vocab | F1 | tok/char |
| :-- | :-- | :-- |
| 1000 | 0.4442 | 0.397 |
| **2000** | **0.4755** | 0.338 |
| 3000 | 0.4459 | 0.313 |
| 5000 | 0.4481 | 0.288 |

**Rejected & why**: regex pre-tok (worse F1), lowercase (loses capitalization entity cue),
noise-norm @/URL/repeat-collapse (changes char count → breaks NER length alignment),
`num_bigrams` sweep (≈flat, ~1% compression).
*(Earlier "tokenizer_3 balanced mix REJECTED" reasoning is now VOID — see Spec update below: `--train_files_3`
makes a custom tok_3 file mix reproducible.)*

**Why domain_1 caps ~0.45–0.48 (data/model ceiling, not a tokenizer bug)**: tiny noisy Twitter data
(3394 sents, 5% entity rate, **67.8% dev entities OOV**, noisy caps). Deeper analysis: subtoken coverage
99.9% and 91.5% of first-subtokens seen as entity-starts in train (vocab-insensitive) — tokenizer already
exposes trainable subwords; bottleneck is the ambiguous capital-start signal + small bi-LSTM. Threshold
later lowered to **0.4** → gate met; domain_1 tuning is now competition-only.

**Competition blend** (F1 + efficiency + speed, equal weight). ⚠️ **Per Q&A (Dvir, 2026-06-23): the three
axes are normalized RELATIVE TO OTHER STUDENTS, then each = 15% of the grade.** So our old "vocab 2000 wins
blend 2.03 vs 1.13" number is meaningless as an absolute — it was normalized over **our own** sweep, not
peers, which we can't see. Decision (vocab 2000) still defensible: best F1 + fast encode, only modest
compression cost. Don't over-optimize a single axis; a reasonable all-round tradeoff is the goal.

## Spec update (2026-06-27) — re-read of spec + Q&A + README
1. **NEW required file `train_commands.txt`** at the **zip root** (not `code/`): exact commands to reproduce
   all 3 tokenizers, incl the `generate_tokenizers.py` call with chosen `--vocab_size` / `--train_files_3`.
   `check_submission.py` only **warns** if missing (soft) — but spec mandates it; format mismatch ⇒ grade 0.
   (This is why our earlier test zip passed without it.)
2. **`--train_files_3` reopens tok_3** (`generate_tokenizers.py:106`,`:80`): per-tokenizer training-file lists
   ARE allowed and we pin our choice in `train_commands.txt`. So a custom tok_3 data strategy is now a
   blessed, reproducible lever (15% + competition). Caveat: CLI only *selects which provided files* — no
   reweight/downsample, and only domain_1/domain_2 train files exist. In-code balancing would perturb tok_1.
   Cheapest safe path: keep naive default `domain_1 + domain_2` and justify in the report.
3. **Threshold 0.4 now official** (spec line 14 + Q&A announcement) — was our assumption.
4. **Bigram rule confirmed** (Q&A, "Ofek Nisan" → 1 token; 3-word token DISQUALIFIES) — our `_is_bigram_surface`
   (≤1 internal space) already enforces.
5. **NER filenames** real on disk: `train_1_binary.tagged` etc. (`data/ner_data/`).
6. **Per-tokenizer vocab ALLOWED** (Q&A, Dvir Jun 28): "If you are using a different vocab size between
   tokenizers, you can provide a command to use `train_tokenizer.py` instead." → we now use 3
   `train_tokenizer.py` commands (d1=2000, d2/d3=5000), `FORCE_VOCAB_SIZE=None`. **Voids** the old
   "one global vocab / per-tokenizer not achievable" note above.
7. **domain_3 file from the two training files ALLOWED** (Q&A, Dvir Jun 29: "Yes") — we pass both files
   to `train_tokenizer.py` for tok_3 (naive concat; balanced mix was tested + rejected earlier).

## Optimization (local CPU, 2026-06-27/28) — current config confirmed + speed win
All experiments run locally on CPU (tokenize ~8 min, NER minutes–30 min; `data/` junctioned so
`--data_dir data` works). **VM restored 2026-06-29** → final reproduction + validation run in the VM uv
env (as required) via `train_commands.txt`; local pkls are byte-identical (deterministic). Dev harnesses:
`_optimize.py` (vocab×nb sweep), `_phase2_gen.py`, `_phase3.py`, `_speedeval.py`, `_vsweep.py` — none submitted.

**vocab × num_bigrams sweep** (full data, full NER, merge encoder):
| vocab | d1 F1 | d2 F1 | d1 tok/char | d2 tok/char |
| :-- | :-- | :-- | :-- | :-- |
| 1500 | 0.4731 | 0.9600 | 0.361 | 0.390 |
| **2000** | **0.4693** | **0.9609** | 0.339 | 0.358 |
| 3000 | 0.4498 | 0.9613 | 0.314 | 0.321 |
Efficiency↔F1 trade; d2 flat ~0.96. 1500 dominated; 3000 = −4% d1 F1 for ~8% compression. **Kept 2000.**
`num_bigrams` 5→200: ~0 compression gain + more entity frag → **kept 5**.

**tok_3 data mix**: naive (d1+d2 full) vs balanced (downsample d1→d2 size). **Naive wins both** proxies
(d1 0.4879 vs 0.4687, d2 0.9642 vs 0.9609) — more data > de-biasing. Balanced helper dropped. **Kept naive.**

**Encoding speed — greedy longest-match ADOPTED** (`encoder="longest"`, default). Cold encode:
| | merge | longest | Δ |
| :-- | :-- | :-- | :-- |
| d1 speed | 363k tok/s | 560k | **+54%** |
| d2 speed | 430k tok/s | 706k | **+64%** |
| d1 tok/char | 0.3394 | 0.3373 | −0.6% |
| d1 F1 | 0.4693 | 0.4597 | −0.0096 |
| d2 F1 | 0.9609 | 0.9599 | −0.0010 |
Big speed gain (a full competition axis) > small F1 dip (both ≫ 0.4); user accepted under equal-weight
peer-normalized competition. Training unchanged (inference-only) → BPE-legal; pkls regenerated, check green.

## Per-tokenizer vocab (local CPU, 2026-06-29) — efficiency win
**New lever** (Q&A: Dvir Jun 28 allows per-tokenizer vocab via `train_tokenizer.py`; Jun 29 allows a
domain_3 file built from the two training files). Voids the old "one global vocab" constraint.

**Diagnostic (why):** domain_1 = 1.15M types, **27% @handles**, 72% hapax → entity F1 is a data ceiling
(keep small vocab). domain_2 = clean, but **596 single-char tokens** (419 ultra-rare unicode) eat the
2000 budget → only 1395 merges, 46% whole-word coverage → big compression headroom at larger vocab. And
**fragmentation ≠ F1** (d2 entities fragment MORE than d1 yet F1 0.96 vs 0.46) → d2/d3 vocab is free to
grow for efficiency.

**Per-domain sweeps** (longest encoder; efficiency = tok/char, lower better; F1 gated):
| | vocab | d1 F1 | d2 F1 | d1 tok/char | d2 tok/char |
| :-- | :-- | :-- | :-- | :-- | :-- |
| domain_1 | **2000** | 0.46 | — | 0.337 | — |
| domain_2 | 2000→**5000** | — | 0.9599→**0.9601** | — | 0.357→**0.283** (−21%) |
| domain_3 | 2000→**5000** | 0.467→**0.443** | 0.963→0.961 | 0.356→**0.294** | 0.384→**0.305** |
d2@5000 = −21% tokens at flat F1 (8000 gives −29% but lower tok/s + F1 0.9588 → chose 5000). d3@5000 =
~−20% tokens both domains (+ faster); d1-proxy F1 dips 0.024 (still ≫0.4) — user chose 5000 for efficiency
over the hidden-F1 risk. **Final: d1=2000, d2=5000, d3=5000.**

## Hard constraints (all satisfied)
1. Char-start BPE, merge upward. 2. Tokens capped at ≤2 words (≤1 internal space) **and ≥1 bigram**
(`check_submission.py`). 3. `space_token` non-None (`▁`) and in `token_to_id`. 4. Reproducible via the
three `train_tokenizer.py` commands in `train_commands.txt` (per-tokenizer vocab; staff-allowed). 5. Pickleable
(cache excluded via `__getstate__`); loadable knowing only `BaseTokenizer`. 6. `decode` round-trips chars +
maps `▁`→space (NER aligns via `decode([id])`). 7. Libs `numpy/regex/torch/tqdm` only; provided data only.
8. Edit only `bpe_tokenizer.py` (+ `train_tokenizer.py`/`test_tokenizer.py`); never touch
`generate_tokenizers.py`, `train_ner_model.py`, `base_tokenizer.py`, `check_submission.py`.
9. **`train_commands.txt` at zip root** (reproducibility) — NEW, mandatory.

## Status
**Valid submission + competition-optimized (per-tokenizer vocab).** Encoder `longest`, nb5,
`FORCE_VOCAB_SIZE=None`. Final pkls: d1=2000, d2=5000, d3=5000 — reproduced via the 3 `train_tokenizer.py`
commands in `train_commands.txt`. Repo: private GitHub `RAZAMZALAG/nlp-hw2-bpe`.

**Done**: word-only code · longest-match encoder (+54–64% speed) · per-tokenizer vocab (d2/d3 efficiency
~−21%) · `train_tokenizer.py` multi-file/output-name · `train_commands.txt` (3 commands) · length-preserving
`[UNK]` · vocab/nb/tok_3/encoder sweeps · diagnostics.

**Remaining TODO**:
- [ ] **Re-build + re-validate**: regenerate the 3 per-vocab pkls (running), rebuild `HW2_206922478.zip`,
  `check_submission.py` green, confirm vocab sizes 2000/5000/5000.
- [ ] **Update report** with per-tokenizer vocab table + efficiency gains; regenerate PDF.
- [ ] **Polished report PDF** — current is an auto-placeholder (fpdf2/Helvetica, `▁`→`_`). Typeset:
  ≤1 A4 page, **Arial 10**, 2.54 cm margins, 1.15 spacing (format mismatch ⇒ grade 0).
- [ ] Revoke the leaked GitHub PAT.

## Grade split
Impl 20% (F1 gate, now 0.4 — official) · Competition 45% = **3 axes × 15% each**, peer-normalized
(efficiency + speed + hidden-domain F1) · Tokenizer-3 eval 15% (report) · Report 20% (strict format).

## How to run
Run from `code_env_example/` (data under `data/`). Locally (no VM/uv) swap `uv run python` → `py` with
`PYTHONUTF8=1`. `encoder="longest"` is the in-code default → baked into the pkls.
```bash
cd code_env_example
# Reproduce the 3 submission tokenizers (these are train_commands.txt verbatim):
uv run python code/train_tokenizer.py --domain_file data/domain_1_train.txt --vocab_size 2000 --output_dir trained_tokenizers --output_name tokenizer_1.pkl
uv run python code/train_tokenizer.py --domain_file data/domain_2_train.txt --vocab_size 5000 --output_dir trained_tokenizers --output_name tokenizer_2.pkl
uv run python code/train_tokenizer.py --domain_file data/domain_1_train.txt data/domain_2_train.txt --vocab_size 5000 --output_dir trained_tokenizers --output_name tokenizer_3.pkl
# Eval / validate:
uv run python code/test_tokenizer.py  --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/domain_1_train.txt --test_file data/domain_1_dev.txt
uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged
uv run python check_submission.py HW2_206922478.zip
```
