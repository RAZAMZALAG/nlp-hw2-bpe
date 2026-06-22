#!/usr/bin/env bash
# domain_1 vocab sweep (cased, word/ws/nb5). Tests: does SMALLER vocab raise OOV recall/F1?
# Competition cares about F1 + tokens/char + speed together. Baseline vocab 5000 = F1 0.4481.
cd ~/HW2/code_env_example
export PATH="$HOME/.local/bin:$PATH"
D=../data/data
for V in 2000 3000 8000; do
  echo "######## domain_1 | cased | vocab=$V"
  uv run python code/_bake.py $D/domain_1_train.txt word $V 0 vd1_$V.pkl 5 ws
  uv run python code/_inspect.py vd1_$V.pkl $D/domain_1_dev.txt 3000 | grep -E "vocab size|tokens/char|speed"
  uv run python code/train_ner_model.py --tokenizer_path vd1_$V.pkl \
    --train_file $D/ner_data/train_1_binary.tagged --dev_file $D/ner_data/dev_1_binary.tagged 2>&1 | grep -E "Best F1"
  echo "## done V=$V"
done
echo "######## VSWEEP DONE"
