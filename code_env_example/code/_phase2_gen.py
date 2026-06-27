"""DEV-ONLY: train sweep tokenizers at given vocabs (nb=5) for Phase-2 NER."""
import sys
import bpe_tokenizer
from bpe_tokenizer import BPETokenizer

bpe_tokenizer.FORCE_VOCAB_SIZE = None
DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"

def load(p):
    with open(p, encoding="utf-8") as f:
        return f.readlines()

for v in (1500, 3000):
    for dom in (1, 2):
        texts = load(f"{DATA}/domain_{dom}_train.txt")
        tok = BPETokenizer(vocab_size=v, num_bigrams=5)
        tok.train(texts)
        out = f"../trained_tokenizers/sweep_d{dom}_v{v}.pkl"
        tok.save(out)
        print(f"saved {out} vocab={tok.get_vocab_size()}", flush=True)
