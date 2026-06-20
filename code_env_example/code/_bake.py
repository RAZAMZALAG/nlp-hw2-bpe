"""Bake-off trainer. Not for submission.
usage: _bake.py <domain> <method> <vocab> <cap> <out.pkl> [num_bigrams] [pretok]"""
import sys, time
from bpe_tokenizer import BPETokenizer

domain, method, vocab, cap, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
nb = int(sys.argv[6]) if len(sys.argv) > 6 else 5
pretok = sys.argv[7] if len(sys.argv) > 7 else "ws"
lower = (sys.argv[8].lower() in ("1", "true", "lower", "lc")) if len(sys.argv) > 8 else False
with open(domain, encoding="utf-8") as f:
    texts = f.readlines()
t0 = time.time()
tk = BPETokenizer(vocab_size=vocab, method=method, num_bigrams=nb, max_train_chars=cap,
                  pretok=pretok, lowercase=lower)
tk.train(texts)
tk.save(out)
print(f"[{method}/{pretok}/nb{nb}] vocab={tk.get_vocab_size()} merges={len(tk.merges)} "
      f"train={time.time()-t0:.0f}s -> {out}", flush=True)
print(f"  top_bigrams={tk.top_bigrams[:5]}", flush=True)
