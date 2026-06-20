#!/usr/bin/env bash
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
D=../data/data
echo "#### d1 | CASED | vocab 10000 (vs cased 5k = 0.4481)"
uv run python code/_bake.py $D/domain_1_train.txt word 10000 0 d1c10.pkl 5 ws
uv run python code/_inspect.py d1c10.pkl $D/domain_1_dev.txt 3000 | grep -E "vocab size|tokens/char|reconstruct"
uv run python code/train_ner_model.py --tokenizer_path d1c10.pkl --train_file $D/ner_data/train_1_binary.tagged --dev_file $D/ner_data/dev_1_binary.tagged 2>&1 | grep -E "Best F1"
echo "#### V10 DONE"
