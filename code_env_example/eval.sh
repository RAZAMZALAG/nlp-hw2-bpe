#!/usr/bin/env bash
# Final VM check: real NER F1 (both domains) + speed/efficiency (all 3 tokenizers).
# Run from code_env_example/:  bash eval.sh
set -e
cd "$(dirname "$0")"

echo "===== NER F1 (full 20-epoch) ====="
for d in 1 2; do
  echo "--- domain $d (tokenizer_$d) ---"
  uv run python code/train_ner_model.py \
    --tokenizer_path trained_tokenizers/tokenizer_${d}.pkl \
    --train_file data/ner_data/train_${d}_binary.tagged \
    --dev_file data/ner_data/dev_${d}_binary.tagged 2>&1 | grep "Best F1"
done

echo "===== Speed + efficiency ====="
test_one () {  # $1=tokenizer id  $2=domain for dev set
  echo "--- tokenizer_$1 on domain_$2 dev ---"
  uv run python code/test_tokenizer.py \
    --tokenizer_path trained_tokenizers/tokenizer_$1.pkl \
    --train_file data/domain_$2_train.txt \
    --test_file data/domain_$2_dev.txt 2>&1 | grep -E "Encoding speed|Tokens per character"
}
test_one 1 1
test_one 2 2
test_one 3 1
test_one 3 2
echo "===== done ====="
