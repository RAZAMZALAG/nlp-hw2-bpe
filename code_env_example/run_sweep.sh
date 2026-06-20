#!/usr/bin/env bash
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
D=../data/data
ev() {  # pkl dev nertrain nerdev
  uv run python code/_inspect.py "$1" "$2" 3000 | grep -E "vocab size|bigram tokens|tokens/char|reconstruct|speed"
  uv run python code/train_ner_model.py --tokenizer_path "$1" --train_file "$3" --dev_file "$4" 2>&1 | grep -E "Best F1"
}
echo "#### A: domain_1 | ws | nb5"
uv run python code/_bake.py $D/domain_1_train.txt word 5000 0 a.pkl 5 ws
ev a.pkl $D/domain_1_dev.txt $D/ner_data/train_1_binary.tagged $D/ner_data/dev_1_binary.tagged
echo "#### B: domain_1 | regex | nb5"
uv run python code/_bake.py $D/domain_1_train.txt word 5000 0 b.pkl 5 regex
ev b.pkl $D/domain_1_dev.txt $D/ner_data/train_1_binary.tagged $D/ner_data/dev_1_binary.tagged
echo "#### C: domain_2 | regex | nb5"
uv run python code/_bake.py $D/domain_2_train.txt word 5000 0 c.pkl 5 regex
ev c.pkl $D/domain_2_dev.txt $D/ner_data/train_2_binary.tagged $D/ner_data/dev_2_binary.tagged
echo "#### SWEEP DONE"
