#!/usr/bin/env bash
# Build a test submission zip and run the course check_submission.py end-to-end.
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
rm -rf sub HW2_123456789.zip HW2_123456789
mkdir -p sub/code sub/trained_tokenizers
cp code/bpe_tokenizer.py sub/code/
cp trained_tokenizers/tokenizer_1.pkl trained_tokenizers/tokenizer_2.pkl trained_tokenizers/tokenizer_3.pkl sub/trained_tokenizers/
printf 'placeholder report' > sub/report_123456789.pdf
uv run python - <<'PY'
import zipfile, os
with zipfile.ZipFile('HW2_123456789.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk('sub'):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, 'sub'))
print('zip built:', os.path.getsize('HW2_123456789.zip'), 'bytes')
PY
# check_submission resolves NER data at ../data/ner_data relative to the extracted dir
# (~/HW2/code_env_example/HW2_123456789/../data) -> symlink that to the real data dir.
ln -sfn ~/HW2/data/data ~/HW2/code_env_example/data
uv run python check_submission.py HW2_123456789.zip
echo "######## CHECKSUB DONE"
