"""DEV-ONLY (not submitted): vocab x num_bigrams sweep.
Measures efficiency (tok/char on dev), reconstruction%, #bigrams, and an entity-fragmentation
proxy (avg subtokens per entity word; lower ~ better first-subtoken recall). No NER here."""
import sys, time
import bpe_tokenizer
from bpe_tokenizer import BPETokenizer
from test_tokenizer import calculate_efficiency, test_reconstruction

bpe_tokenizer.FORCE_VOCAB_SIZE = None  # let the sweep control vocab

DATA = "../data/data" if len(sys.argv) < 2 else sys.argv[1]

def load(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]

def entity_words(tagged):
    ws = []
    with open(tagged, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2 and parts[1] == "1":
                ws.append(parts[0])
    return ws

def frag_proxy(tok, words):
    if not words:
        return 0.0
    tot = sum(len(tok._bpe_word(list("▁" + w))) for w in words)
    return tot / len(words)

DOMAINS = {
    1: (f"{DATA}/domain_1_train.txt", f"{DATA}/domain_1_dev.txt", f"{DATA}/ner_data/train_1_binary.tagged"),
    2: (f"{DATA}/domain_2_train.txt", f"{DATA}/domain_2_dev.txt", f"{DATA}/ner_data/train_2_binary.tagged"),
}
GRID_V = [1500, 2000, 3000]
GRID_N = [5, 50, 200]

print(f"{'dom':>3} {'vocab':>5} {'nb':>4} {'tok/char':>9} {'recon%':>7} {'#bg':>4} {'frag(ent)':>9} {'train_s':>8}")
for d, (trainf, devf, taggedf) in DOMAINS.items():
    train_texts = load(trainf)
    dev_texts = load(devf)
    ents = entity_words(taggedf)
    for v in GRID_V:
        for n in GRID_N:
            t0 = time.time()
            tok = BPETokenizer(vocab_size=v, num_bigrams=n)
            tok.train(train_texts)
            dt = time.time() - t0
            eff = calculate_efficiency(tok, train_texts, dev_texts)
            rec = test_reconstruction(tok, dev_texts, sample_size=200) * 100
            nbg = sum(1 for k in tok.token_to_id
                      if k not in tok.special_tokens and " " in k.replace("▁", " ").strip())
            fr = frag_proxy(tok, ents)
            print(f"{d:>3} {v:>5} {n:>4} {eff:>9.4f} {rec:>7.1f} {nbg:>4} {fr:>9.3f} {dt:>8.1f}",
                  flush=True)
