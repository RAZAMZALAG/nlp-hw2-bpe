"""Pull report numbers from the submitted tokenizers. CPU-only."""
import time
from base_tokenizer import BaseTokenizer

DD = "../../data/data"
devs = {1: f"{DD}/domain_1_dev.txt", 2: f"{DD}/domain_2_dev.txt"}


def eff(tk, path, n=3000):
    lines = [l.strip() for l in open(path, encoding="utf-8")][:n]
    tk._cache.clear() if hasattr(tk, "_cache") else None
    t0 = time.time()
    tot = chars = 0
    for _ in range(3):
        for s in lines:
            tot += len(tk.encode(s)); chars += len(s)
    dt = time.time() - t0
    return tot / chars, tot / (3 * len(lines)), tot / dt


for i in (1, 2, 3):
    tk = BaseTokenizer.load(f"../trained_tokenizers/tokenizer_{i}.pkl")
    print(f"\n===== tokenizer_{i}  (vocab {tk.get_vocab_size()}, space {tk.space_token!r}) =====")
    print("top-5 bigrams:   ", getattr(tk, "top_bigrams", [])[:5])
    print("least-5 bigrams: ", getattr(tk, "least_bigrams", [])[:5])
    for d in ([1] if i == 1 else [2] if i == 2 else [1, 2]):
        tc, ts, sp = eff(tk, devs[d])
        print(f"  domain_{d} dev: tokens/char={tc:.4f}  tokens/sent={ts:.2f}  enc={sp:,.0f} tok/s")
