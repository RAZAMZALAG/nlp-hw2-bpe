"""Trainable-subword coverage for OOV entities: can the model learn them via shared subtokens?
Compares vocab 2000 vs 5000. CPU-only."""
from bpe_tokenizer import BPETokenizer

DD = "../../data/data"
ND = f"{DD}/ner_data"


def read(f):
    W, L = [], []
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.strip():
            p = line.split("\t")
            if len(p) == 2:
                W.append(p[0]); L.append(1 if p[1] != "0" else 0)
    return W, L


trW, trL = read(f"{ND}/train_1_binary.tagged")
dvW, dvL = read(f"{ND}/dev_1_binary.tagged")
train_seen_word = set(trW)

train_txt = open(f"{DD}/domain_1_train.txt", encoding="utf-8").readlines()


def toks(tk, w):
    return tk._bpe_word(list("▁" + w))  # mid-sentence surface tokens


for V in (2000, 5000):
    print(f"\n===== vocab {V} =====", flush=True)
    tk = BPETokenizer(vocab_size=V, method="word", num_bigrams=5, max_train_chars=8_000_000)
    tk.train(train_txt)

    seen_sub = set()                 # every subtoken seen anywhere in NER train
    seen_ent_first = set()           # first-subtoken of ENTITY words in NER train
    for w, l in zip(trW, trL):
        ts = toks(tk, w)
        seen_sub.update(ts)
        if l:
            seen_ent_first.add(ts[0])

    dv_ent_oov = [w for w, l in zip(dvW, dvL) if l and w not in train_seen_word]
    cov, first_seen, first_ent = 0.0, 0, 0
    for w in dv_ent_oov:
        ts = toks(tk, w)
        cov += sum(t in seen_sub for t in ts) / len(ts)
        first_seen += ts[0] in seen_sub
        first_ent += ts[0] in seen_ent_first
    n = len(dv_ent_oov)
    print(f"OOV dev entities n={n}")
    print(f"  avg subtoken coverage (subtoks seen in train): {100*cov/n:.1f}%")
    print(f"  first-subtoken seen in train at all:           {100*first_seen/n:.1f}%")
    print(f"  first-subtoken seen as an ENTITY-start in train:{100*first_ent/n:.1f}%")
    print(f"  |seen_ent_first| distinct entity-start tokens = {len(seen_ent_first)}")
