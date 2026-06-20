#!/usr/bin/env bash
set -e
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
D=../data/data
NER_T=$D/ner_data/train_1_binary.tagged
NER_D=$D/ner_data/dev_1_binary.tagged

run() {  # method cap outpkl
  echo "======================== METHOD=$1 ========================"
  uv run python code/_bake.py $D/domain_1_train.txt "$1" 5000 "$2" "$3"
  uv run python code/_inspect.py "$3" $D/domain_1_dev.txt 3000
  uv run python code/train_ner_model.py --tokenizer_path "$3" --train_file "$NER_T" --dev_file "$NER_D" 2>&1 | grep -E "Epoch|Best F1|Dev F1"
}

run word 0       tok_word.pkl
run byte 2000000 tok_byte.pkl
echo "======================== BAKEOFF DONE ========================"
