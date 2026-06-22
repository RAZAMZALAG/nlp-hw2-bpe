"""Check the first-subtoken signal for entity words (the NER label lands on the first subtoken).
Hypothesis: OOV entity words often start with the bare marker '▁' -> non-discriminative -> low recall."""
from collections import Counter
from bpe_tokenizer import BPETokenizer

DD = "../../data/data"
ND = f"{DD}/ner_data"

print("training cased word/ws tokenizer (8MB sample, vocab 5000)...", flush=True)
train = open(f"{DD}/domain_1_train.txt", encoding="utf-8").readlines()
tk = BPETokenizer(vocab_size=5000, method="word", num_bigrams=5, max_train_chars=8_000_000)
tk.train(train)

# collect dev_1 words by label + whether OOV vs tokenizer training words is hard; use NER-train seen set
def read(f):
    W, L = [], []
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        p = line.split("\t")
        if len(p) == 2:
            W.append(p[0]); L.append(1 if p[1] != "0" else 0)
    return W, L

trW, trL = read(f"{ND}/train_1_binary.tagged")
dvW, dvL = read(f"{ND}/dev_1_binary.tagged")
train_seen = set(trW)

def first_tok(word):
    ids = tk.encode(" " + word)  # mid-sentence form
    return tk.id_to_token[ids[0]], len(ids)

def report(tag, words):
    bare = 0; firsts = Counter(); subs = 0
    for w in words:
        ft, n = first_tok(w)
        subs += n
        firsts[ft] += 1
        if ft == tk.space_token:  # bare marker '▁'
            bare += 1
    print(f"\n[{tag}] n={len(words)}  avg_subtokens={subs/len(words):.2f}  "
          f"first-token==bare '▁': {100*bare/len(words):.1f}%")
    print("  top first-tokens:", firsts.most_common(10))

ent = [w for w, l in zip(dvW, dvL) if l]
non = [w for w, l in zip(dvW, dvL) if not l]
ent_oov = [w for w, l in zip(dvW, dvL) if l and w not in train_seen]

report("DEV entity words", ent)
report("DEV entity words OOV (not in NER train)", ent_oov)
report("DEV non-entity words", non)
