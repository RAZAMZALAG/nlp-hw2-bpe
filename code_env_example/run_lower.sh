#!/usr/bin/env bash
# Test the lowercase lever on GPU. Baseline (d1 cased 5k) = 0.4481 already known.
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
D=../data/data
ev() {  # pkl dev nertrain nerdev
  uv run python code/_inspect.py "$1" "$2" 3000 | grep -E "vocab size|tokens/char|reconstruct"
  uv run python code/train_ner_model.py --tokenizer_path "$1" --train_file "$3" --dev_file "$4" 2>&1 | grep -E "Best F1"
}
echo "#### d1 | lowercase | 5000   (vs cased 0.4481)"
uv run python code/_bake.py $D/domain_1_train.txt word 5000 0 d1lc.pkl 5 ws lower
ev d1lc.pkl $D/domain_1_dev.txt $D/ner_data/train_1_binary.tagged $D/ner_data/dev_1_binary.tagged
echo "#### d2 | lowercase | 5000   (must stay >=0.5; cased was 0.96)"
uv run python code/_bake.py $D/domain_2_train.txt word 5000 0 d2lc.pkl 5 ws lower
ev d2lc.pkl $D/domain_2_dev.txt $D/ner_data/train_2_binary.tagged $D/ner_data/dev_2_binary.tagged
echo "#### d1 | lowercase | 10000  (diagnostic only - vocab policy blocked)"
uv run python code/_bake.py $D/domain_1_train.txt word 10000 0 d1lc10.pkl 5 ws lower
ev d1lc10.pkl $D/domain_1_dev.txt $D/ner_data/train_1_binary.tagged $D/ner_data/dev_1_binary.tagged
echo "#### LOWER SWEEP DONE"
