#!/usr/bin/env bash
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
D=../data/data
for pk in tok_word tok_byte; do
  echo "======================== $pk ========================"
  uv run python code/_inspect.py "$pk.pkl" "$D/domain_1_dev.txt" 3000
  uv run python code/train_ner_model.py --tokenizer_path "$pk.pkl" \
    --train_file "$D/ner_data/train_1_binary.tagged" \
    --dev_file "$D/ner_data/dev_1_binary.tagged" 2>&1 | grep -E "Best F1|Dev F1:"
done
echo "======================== EVAL DONE ========================"
