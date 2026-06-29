"""DEV-ONLY: train one domain at several vocabs (nb5, longest), save + measure tok/char + cold speed.
Usage (from code_env_example/):  py code/_vsweep.py <domain: 1|2|3> <vocab,vocab,...>
domain 3 = naive domain_1+domain_2 concat; measured on both dev sets."""
import sys, time
import bpe_tokenizer
from bpe_tokenizer import BPETokenizer
bpe_tokenizer.FORCE_VOCAB_SIZE = None

def load(p):
    with open(p, encoding="utf-8") as f:
        return f.readlines()

def devtexts(p):
    return [l.strip() for l in load(p)]

def measure(tok, dev):
    chars = sum(len(t) for t in dev)
    t0 = time.time(); ntok = 0
    for t in dev:
        ntok += len(tok.encode(t))
    dt = time.time() - t0
    return ntok / chars, ntok / dt

dom = sys.argv[1]
vocabs = [int(x) for x in sys.argv[2].split(",")]
if dom == "3":
    train = load("data/domain_1_train.txt") + load("data/domain_2_train.txt")
    devs = [("d1", devtexts("data/domain_1_dev.txt")), ("d2", devtexts("data/domain_2_dev.txt"))]
    tag = "d3naive"
else:
    train = load(f"data/domain_{dom}_train.txt")
    devs = [(f"d{dom}", devtexts(f"data/domain_{dom}_dev.txt"))]
    tag = f"d{dom}"

for v in vocabs:
    t0 = time.time()
    tok = BPETokenizer(vocab_size=v, num_bigrams=5)
    tok.train(train)
    out = f"trained_tokenizers/sweep_{tag}_v{v}.pkl"
    tok.save(out)
    res = " | ".join(f"{name} tok/char {e:.4f} cold {s:.0f}t/s" for name, (e, s) in
                      ((n, (measure(tok, dv))) for n, dv in devs))
    print(f"{tag} v{v} (train {time.time()-t0:.0f}s, real_vocab {tok.get_vocab_size()}) :: {res}", flush=True)
