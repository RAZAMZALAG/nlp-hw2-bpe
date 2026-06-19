# NLP HW2 – BPE Tokenizer

## How to talk to me
When explaining concepts, always use examples.
Bad: "BPE merges the most frequent pair."
Good: "BPE merges the most frequent pair — e.g., if 'e' and 's' appear together 200 times, they become 'es' in the vocab."

## Project goal
Implement a BPE tokenizer in `code_env_example/code/bpe_tokenizer.py`, train it on domain data, and evaluate on a downstream NER task.

## The only file to implement
`code_env_example/code/bpe_tokenizer.py` — fill in `BPETokenizer.train()`, `.encode()`, `.decode()`.

## Hard requirements (from spec + check_submission.py)
1. **True BPE**: start from individual characters, merge upward — not word-level.
2. **At least one bigram token**: a single vocab token that spans two whitespace-separated words (e.g., `"New▁York"` where `▁` is the space token). Without this, submission is disqualified.
3. **`self.space_token` must not be `None`** — the NER pipeline and submission checker both reject `None`.
4. **Only provided data** may be used for training (files under `data/`).
5. **Allowed libraries**: only `numpy`, `regex`, `torch`, `tqdm` (from `pyproject.toml`). No HuggingFace tokenizers or other external tokenizer libs.

## Files to NEVER edit
- `code_env_example/generate_tokenizers.py`
- `code_env_example/check_submission.py`
- NER hyperparameters inside `code_env_example/code/train_ner_model.py`:
  - `SEED = 42`, `BATCH_SIZE = 32`, `LEARNING_RATE = 0.01`, `NUM_EPOCHS = 20`

## Submission structure (zip must pass check_submission.py)
```
HW2_<id>.zip
├── code/
│   └── bpe_tokenizer.py
├── trained_tokenizers/
│   ├── tokenizer_1.pkl   ← trained on domain_1
│   ├── tokenizer_2.pkl   ← trained on domain_2
│   └── tokenizer_3.pkl   ← trained on hidden domain (via generate_tokenizers.py)
└── report_<id>.pdf
```

## How to run
```bash
# Setup (once)
bash code_env_example/init.sh

# Work from code_env_example/
cd code_env_example

# Train one tokenizer manually
uv run python code/train_tokenizer.py --domain_file ../data/data/domain_1_train.txt --output_dir tokenizers --vocab_size 5000

# Generate all 3 tokenizers reproducibly (use before submission)
uv run python generate_tokenizers.py

# Test tokenizer manually
uv run python code/test_tokenizer.py --tokenizer_path tokenizers/tokenizer.pkl --train_file ../data/data/domain_1_train.txt --test_file ../data/data/domain_1_dev.txt

# Train NER model
uv run python code/train_ner_model.py --tokenizer_path tokenizers/tokenizer.pkl --train_file ../data/ner_data/train_1_binary.tagged --dev_file ../data/ner_data/dev_1_binary.tagged

# Validate zip before submitting
uv run python check_submission.py HW2_<id>.zip
```

## Data files
```
data/data/domain_1_train.txt   ← train tokenizer_1
data/data/domain_2_train.txt   ← train tokenizer_2
data/data/domain_1_dev.txt
data/data/domain_2_dev.txt
data/data/ner_data/train_1_binary.tagged
data/data/ner_data/dev_1_binary.tagged
data/data/ner_data/train_2_binary.tagged
data/data/ner_data/dev_2_binary.tagged
```

## Evaluation criteria
1. **Correctness** — proper BPE algorithm
2. **Speed** — encoding tokens/sec
3. **Efficiency** — compression on unseen text (fewer tokens = better)
4. **NER F1** — downstream task with bi-LSTM model (fixed hyperparams above)

## NER alignment note
The NER pipeline uses "first-subtoken labeling":
- Word "Jobs" splits into tokens `["Jo", "bs"]` → only `"Jo"` gets the entity label, `"bs"` gets `-100` (ignored).
- `space_token` tells the NER pipeline where word boundaries are.

## Reference material
- `Lecture 07 - Tokenization.pptx.pdf` — BPE algorithm theory
- `Tutorial 7 - Colab.pdf` — implementation walkthrough
- `HW2 26.pdf` — full assignment spec (always check this for constraints before implementing)
