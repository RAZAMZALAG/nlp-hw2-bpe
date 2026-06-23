"""Competition-blend sweep (CPU): efficiency (tokens/char, tokens/sentence) + encode speed
across vocab x num_bigrams. F1 is near-ceiling (>=0.4) so measured separately on GPU. Not for submission."""
import time
from bpe_tokenizer import BPETokenizer

DD = "../../data/data"
train = open(f"{DD}/domain_1_train.txt", encoding="utf-8").readlines()
dev = [l.strip() for l in open(f"{DD}/domain_1_dev.txt", encoding="utf-8")][:3000]

configs = [(v, 5) for v in (2000, 3000, 4000, 5000)] + [(3000, 50), (3000, 200)]

print(f"{'vocab':>6} {'nb':>4} | {'bigrams':>7} | {'tok/char':>8} | {'tok/sent':>8} | {'enc tok/s':>10}")
print("-" * 60)
for V, NB in configs:
    tk = BPETokenizer(vocab_size=V, method="word", num_bigrams=NB, max_train_chars=8_000_000)
    tk.train(train)
    specials = set(tk.special_tokens)
    bg = sum(1 for t in tk.token_to_id
             if t not in specials and " " in t.replace(tk.space_token, " ").strip())
    # efficiency + speed
    tk._cache.clear()
    t0 = time.time()
    total_tok = total_chr = 0
    for _ in range(3):
        for s in dev:
            ids = tk.encode(s)
            total_tok += len(ids); total_chr += len(s)
    dt = time.time() - t0
    tok_char = total_tok / total_chr
    tok_sent = total_tok / (3 * len(dev))
    speed = total_tok / dt
    print(f"{V:>6} {NB:>4} | {bg:>7} | {tok_char:>8.4f} | {tok_sent:>8.2f} | {speed:>10,.0f}", flush=True)
