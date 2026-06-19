# HW2 — BPE Tokenizer + NER Evaluation: Project Overview

## 1. The one-sentence task
Implement a **Byte Pair Encoding (BPE) tokenizer** from scratch (chars → merge up), train 3 versions
on provided text, and have them judged by how well a **fixed bi-LSTM NER model** performs with them.

## 2. The ONLY file you implement
`code_env_example/code/bpe_tokenizer.py` — class `BPETokenizer(BaseTokenizer)`.
Fill 3 methods, currently `raise NotImplementedError`:
- `train(self, texts: List[str]) -> None`
- `encode(self, text: str) -> List[int]`
- `decode(self, token_ids: List[int]) -> str`

You may add helper modules in `code/`, but `bpe_tokenizer.py` must hold `BPETokenizer`
(both `train_tokenizer.py` and `generate_tokenizers.py` import it by that name).

## 3. What you inherit (`code/base_tokenizer.py` — DON'T edit)
- `__init__` pre-fills special tokens: `[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3`. Your real tokens start at id 4.
- Maintains `self.token_to_id` / `self.id_to_token` (you populate these in `train`).
- Gives you `encode_batch`, `decode_batch`, `get_vocab_size`, `save` (pickle), `load`.
- **Pickle**: the whole object is pickled. Every attribute you set must be pickleable (no lambdas,
  no open file handles, no non-picklable compiled objects). Plain dicts/lists/strings are fine.

## 4. HARD requirements (violate → disqualified). Enforced by `check_submission.py`.
1. **True BPE** — start from individual characters, iteratively merge the most-frequent adjacent pair.
   NOT word-level. e.g. count pairs; if `t`+`h` is most frequent, merge to `th`; repeat until vocab full.
2. **`self.space_token` must NOT be `None`.** Currently `= None` in the stub. Set it (e.g. `'▁'`).
   - Checker reads `getattr(tokenizer,'space_token')`; `None` → fail (`check_submission.py:254`).
   - For the NER pipeline to use it, the space_token should also be a real entry in `token_to_id`
     (`train_ner_model.py:49` looks up `tokenizer.space_token in tokenizer.token_to_id`).
3. **At least one BIGRAM token** — a single vocab token that, after turning `space_token` back into
   a real space, contains an internal space between two words (e.g. `"New▁York"` → `"New York"`).
   - Exact check (`check_submission.py:263-277`): for each non-special token, `tok.replace(space_token,' ')`,
     then `' ' in surface.strip()`. ≥1 must pass. So you MUST let merges cross the space boundary at
     least once. Easiest: include `space_token` as a mergeable symbol so pairs like `New`+`▁York` form.
4. **Only the provided data** under `data/` may be used for training. No external corpora.
5. **Allowed libs only**: `numpy`, `regex`, `torch`, `tqdm` (`pyproject.toml`). No HuggingFace / other tokenizers.

## 5. CRITICAL design constraint from the NER pipeline (read before coding `decode`)
`train_ner_model.py` (`NERDataset`, lines 117-154) aligns tokens to words by **calling
`tokenizer.decode([single_id])` and incremental `decode(token_ids[:i+1])`** to find each token's
character span in the original text. Implications:
- `decode` of the space token should produce a **real space `" "`**, and decode must be roughly
  invertible at the character level so cumulative decoded length tracks the original text.
- NER input text is words joined by single spaces (`read_ner_data`: `' '.join(words)`), labels are
  per-word binary. **First-subtoken labeling**: only the first token of each word gets the label;
  other subtokens get `-100` (ignored in loss). README example:
  "Steve Jobs" → tokens `["Steve J","ob","s ate",...]`; only `"Steve J"` gets label 1.
- Reconstruction test (`test_tokenizer.py:90`) is lenient: compares `decoded.replace(" ","")` vs
  `original.replace(" ","")`. So whitespace differences are forgiven, but characters must round-trip.

## 6. The three tokenizers (built by `generate_tokenizers.py` — DON'T edit; default vocab_size=5000)
| File | Trained on (default) | Judged on |
|---|---|---|
| `tokenizer_1.pkl` | `domain_1_train.txt` | NER domain 1 (Twitter) |
| `tokenizer_2.pkl` | `domain_2_train.txt` | NER domain 2 (News) |
| `tokenizer_3.pkl` | `domain_1_train.txt` + `domain_2_train.txt` (naive baseline) | **hidden** domain — strategy is graded |

`tokenizer_3` strategy is open: the hidden domain differs from both, so a smarter data mix may beat
naive concat. Override the mix via `--train_files_3`.

## 7. Data (at repo root `data/data/`, NOT `code_env_example/data/`)
| File | Lines | Content |
|---|---|---|
| `domain_1_train.txt` / `_dev.txt` | 1.28M / 160K | **Twitter**: informal, typos, `@mentions`, slang. e.g. `"Twit twit twit. Time to crean my room..."` |
| `domain_2_train.txt` / `_dev.txt` | 320K / 40K | **News**: formal, long sentences. e.g. `"Funky and fascinating New Orleans has seen its share of hard times..."` |
| `ner_data/train_{1,2}_binary.tagged` | 66K / 220K | NER train, `word<TAB>label` |
| `ner_data/dev_{1,2}_binary.tagged` | 17K / 55K | NER eval |

`.tagged` format: one `word<TAB>tag` per line; tag `0`=non-entity, anything else → entity (binary 1);
blank line = sentence boundary. Example:
```
Empire	1
State	1
Building	1
=	0
ESB	1
```

## 8. ⚠️ Path gotcha (will bite immediately)
`generate_tokenizers.py` defaults `--data_dir data` and is meant to run from `code_env_example/`.
But the actual data is at repo-root `data/data/`. So from `code_env_example/` you must run:
```bash
uv run python generate_tokenizers.py --data_dir ../data/data
```
The README's `data/domain_1_train.txt` paths assume a layout that doesn't match this checkout.
The `CLAUDE.md` `train_tokenizer.py` example already uses the correct `../data/data/...`.

## 9. Provided scripts (DON'T edit any of these)
- `generate_tokenizers.py` — trains & saves all 3 `.pkl` reproducibly (seed 42).
- `check_submission.py` — validates the zip: structure, loads each tokenizer via `BaseTokenizer.load`,
  runs `encode("Hello world!")/decode`, checks `space_token` + bigram, runs a 1-batch NER smoke test.
- `code/train_tokenizer.py` — train ONE tokenizer → `tokenizer.pkl`.
- `code/test_tokenizer.py` — prints speed (tokens/sec), efficiency (tokens/char), reconstruction %, samples.
- `code/train_ner_model.py` — bi-LSTM NER. **Locked hyperparams**: `SEED=42, BATCH_SIZE=32, LR=0.01, NUM_EPOCHS=20`.

## 10. How you're graded (4 axes)
1. **Correctness** — genuine BPE, passes hard requirements, sane F1 (target ≥0.5 on domain dev sets).
2. **Speed** — `encode` tokens/sec.
3. **Efficiency** — tokens/char on unseen text (lower = better compression).
4. **NER F1** — downstream bi-LSTM performance (the real competition signal, incl. hidden domain).

Trade-off: tiny vocab = fast but poor F1; huge vocab = better F1 but slower / less general. You tune this.
(Exact % weights are in `HW2 26.pdf` — confirm there before finalizing the report.)

## 11. Submission format (filename → checker parses IDs)
```
HW2_<id>.zip            (or HW2_<id1>_<id2>.zip)
├── code/
│   └── bpe_tokenizer.py        (+ any helper .py)
├── trained_tokenizers/
│   ├── tokenizer_1.pkl
│   ├── tokenizer_2.pkl
│   └── tokenizer_3.pkl
└── report_<id>.pdf             (must match the ID(s) in the zip name)
```
Report (per spec/PDF): training method per tokenizer, hidden-domain strategy, top-5 most/least
frequent bigrams, F1 + efficiency on dev sets. Confirm exact formatting rules in `HW2 26.pdf`.

## 12. Typical workflow
```bash
bash code_env_example/init.sh          # once: installs uv, builds env
cd code_env_example
# train one quickly:
uv run python code/train_tokenizer.py --domain_file ../data/data/domain_1_train.txt --output_dir tokenizers --vocab_size 5000
# inspect:
uv run python code/test_tokenizer.py --tokenizer_path tokenizers/tokenizer.pkl --train_file ../data/data/domain_1_train.txt --test_file ../data/data/domain_1_dev.txt
# downstream NER:
uv run python code/train_ner_model.py --tokenizer_path tokenizers/tokenizer.pkl --train_file ../data/data/ner_data/train_1_binary.tagged --dev_file ../data/data/ner_data/dev_1_binary.tagged
# all 3 for submission:
uv run python generate_tokenizers.py --data_dir ../data/data
# validate zip:
uv run python check_submission.py HW2_<id>.zip
```

## 13. First implementation gotchas (for when we start)
- 1.28M lines (91MB) for domain 1 — naive O(vocab·corpus) pair counting will be slow. Plan for
  efficient pair counts (count once, update incrementally, or word-frequency dict + `tqdm`).
- Decide space representation up front (`▁` GPT-style prefix vs explicit space token) — it drives
  both the bigram requirement AND the NER decode alignment.
- Reserve ids 0-3 for special tokens; unknown chars at `encode` time → `[UNK]`=1.

---
*All claims verified directly against the source files (`base_tokenizer.py`, `bpe_tokenizer.py`,
`train_tokenizer.py`, `test_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`,
`check_submission.py`, `CLAUDE.md`, `README.md`) and the data files. Items marked "confirm in PDF"
(exact grade %, report formatting) come only from `HW2 26.pdf`.*
