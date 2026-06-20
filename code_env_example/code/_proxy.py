"""CPU-only proxy to rank configs before spending GPU/F1. Not for submission.
Fewer subtokens per ENTITY word => its first-subtoken label covers more of the word => recall proxy."""
import time
from bpe_tokenizer import BPETokenizer

D = "../../data/data"
train = open(f"{D}/domain_1_train.txt", encoding="utf-8").readlines()
dev = [l.strip() for l in open(f"{D}/domain_1_dev.txt", encoding="utf-8")][:3000]

ent, allw = [], []
for line in open(f"{D}/ner_data/train_1_binary.tagged", encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    w, t = parts
    allw.append(w)
    if t != "0":
        ent.append(w)
print(f"entity-word occurrences={len(ent)} total-words={len(allw)}", flush=True)


def avg_sub(tok, words):
    return sum(len(tok.encode(" " + w)) for w in words) / len(words)


def tpc(tok, lines):
    tt = sum(len(tok.encode(s)) for s in lines)
    tc = sum(len(s) for s in lines)
    return tt / tc if tc else 0


for vocab in (5000, 10000):
    for lower in (False, True):
        t0 = time.time()
        tk = BPETokenizer(vocab_size=vocab, method="word", num_bigrams=5,
                          max_train_chars=8_000_000, lowercase=lower)
        tk.train(train)
        print(f"vocab={vocab} lower={lower} | tok/char={tpc(tk, dev):.4f} | "
              f"sub/word(all)={avg_sub(tk, allw):.3f} | sub/word(ENTITY)={avg_sub(tk, ent):.3f} | "
              f"train={time.time()-t0:.0f}s", flush=True)
