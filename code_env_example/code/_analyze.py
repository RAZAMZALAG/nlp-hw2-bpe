"""Data analysis for domain_1 vs domain_2 NER. Why is domain_1 F1 low? CPU-only."""
from collections import Counter

D = "../../data/data/ner_data"


def read(tag_file):
    sents, words, tags = [], [], Counter()
    cur_w, cur_l = [], []
    for line in open(tag_file, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur_w:
                sents.append((cur_w, cur_l)); cur_w, cur_l = [], []
            continue
        p = line.split("\t")
        if len(p) != 2:
            continue
        w, t = p
        cur_w.append(w); cur_l.append(1 if t != "0" else 0)
        tags[t] += 1
    if cur_w:
        sents.append((cur_w, cur_l))
    return sents, tags


def ent_spans(labels):
    """Count entity SPANS (runs of 1s) and their lengths."""
    spans, n = [], 0
    for l in labels:
        if l == 1:
            n += 1
        elif n:
            spans.append(n); n = 0
    if n:
        spans.append(n)
    return spans


def analyze(name, train_f, dev_f):
    tr, tags = read(train_f)
    dv, _ = read(dev_f)
    tr_words = [w for s, _ in tr for w in s]
    tr_labels = [l for _, ls in tr for l in ls]
    dv_words = [w for s, _ in dv for w in s]
    dv_labels = [l for _, ls in dv for l in ls]

    tr_ent = [w for w, l in zip(tr_words, tr_labels) if l]
    dv_ent = [w for w, l in zip(dv_words, dv_labels) if l]

    spans = [s for _, ls in tr for s in ent_spans(ls)]
    multi = sum(1 for s in spans if s > 1)

    def caps(ws): return 100 * sum(w[:1].isupper() for w in ws) / max(1, len(ws))
    def at(ws): return 100 * sum(w.startswith(("@", "#")) for w in ws) / max(1, len(ws))
    def avglen(ws): return sum(len(w) for w in ws) / max(1, len(ws))

    tr_ent_set = set(tr_ent)
    tr_word_set = set(tr_words)
    dv_ent_unseen_as_ent = 100 * sum(w not in tr_ent_set for w in dv_ent) / max(1, len(dv_ent))
    dv_ent_unseen_at_all = 100 * sum(w not in tr_word_set for w in dv_ent) / max(1, len(dv_ent))

    print(f"\n========== {name} ==========")
    print(f"raw tag values: {dict(tags)}")
    print(f"TRAIN: sentences={len(tr)}  words={len(tr_words)}  entity-words={len(tr_ent)} "
          f"({100*len(tr_ent)/len(tr_words):.2f}%)")
    print(f"DEV:   sentences={len(dv)}  words={len(dv_words)}  entity-words={len(dv_ent)} "
          f"({100*len(dv_ent)/len(dv_words):.2f}%)")
    print(f"entity spans (train): total={len(spans)}  multi-word={multi} "
          f"({100*multi/max(1,len(spans)):.1f}%)  avg-span-len={sum(spans)/max(1,len(spans)):.2f}")
    print(f"entity words: capitalized={caps(tr_ent):.1f}%  @/#={at(tr_ent):.1f}%  avg-len={avglen(tr_ent):.1f}")
    print(f"non-entity:   capitalized={caps([w for w,l in zip(tr_words,tr_labels) if not l]):.1f}%")
    print(f"unique entity words (train)={len(tr_ent_set)}")
    print(f"DEV entity words NOT seen as entity in train: {dv_ent_unseen_as_ent:.1f}%")
    print(f"DEV entity words NOT seen at all in train:    {dv_ent_unseen_at_all:.1f}%")
    print("top-15 train entity words:", Counter(tr_ent).most_common(15))


analyze("DOMAIN_1 (Twitter)", f"{D}/train_1_binary.tagged", f"{D}/dev_1_binary.tagged")
analyze("DOMAIN_2 (News)", f"{D}/train_2_binary.tagged", f"{D}/dev_2_binary.tagged")
