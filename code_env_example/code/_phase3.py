"""DEV-ONLY: train balanced tok_3 (vocab 2000, nb5) + efficiency vs naive tok_3. Run from code_env_example/."""
import bpe_tokenizer
from bpe_tokenizer import BPETokenizer
from base_tokenizer import BaseTokenizer
from test_tokenizer import calculate_efficiency

bpe_tokenizer.FORCE_VOCAB_SIZE = None

def load(p):
    with open(p, encoding="utf-8") as f:
        return f.readlines()

# train balanced tok_3
texts = load("data/domain_3_balanced.txt")
tok = BPETokenizer(vocab_size=2000, num_bigrams=5)
tok.train(texts)
tok.save("trained_tokenizers/tok3_balanced.pkl")
print("saved trained_tokenizers/tok3_balanced.pkl vocab", tok.get_vocab_size(), flush=True)

d1 = [l.strip() for l in load("data/domain_1_dev.txt")]
d2 = [l.strip() for l in load("data/domain_2_dev.txt")]
print(f"{'variant':>10} {'d1 tok/char':>12} {'d2 tok/char':>12}")
for name, path in [("naive", "trained_tokenizers/tokenizer_3.pkl"),
                   ("balanced", "trained_tokenizers/tok3_balanced.pkl")]:
    t = BaseTokenizer.load(path)
    print(f"{name:>10} {calculate_efficiency(t,[],d1):>12.4f} {calculate_efficiency(t,[],d2):>12.4f}", flush=True)
