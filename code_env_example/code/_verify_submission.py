"""Verify the 3 generated tokenizers meet the hard requirements (no torch / no NER)."""
from base_tokenizer import BaseTokenizer

for i in (1, 2, 3):
    p = f"../trained_tokenizers/tokenizer_{i}.pkl"
    tk = BaseTokenizer.load(p)
    sp = getattr(tk, "space_token", None)
    specials = set(getattr(tk, "special_tokens", {}))

    # encode/decode smoke (what check_submission runs)
    s = "Hello world! New York is great."
    ids = tk.encode(s)
    dec = tk.decode(ids)

    # bigram + <=2-word cap over the vocab
    bigrams, over = [], []
    for t in tk.token_to_id:
        if t in specials:
            continue
        surface = t.replace(sp, " ") if sp else t
        n = surface.strip().count(" ")
        if n == 1:
            bigrams.append(t)
        elif n >= 2:
            over.append(t)

    print(f"tokenizer_{i}: vocab={tk.get_vocab_size()} space_token={sp!r} "
          f"in_vocab={sp in tk.token_to_id} bigrams={len(bigrams)} over_2word={len(over)}")
    print(f"   e.g. bigram={bigrams[0]!r}" if bigrams else "   NO BIGRAM!")
    print(f"   recon(no-space)={dec.replace(' ','')==s.replace(' ','')}  decode={dec!r}")
    assert sp and sp in tk.token_to_id, "space_token bad"
    assert bigrams, "no bigram -> disqualified"
    assert not over, f"token >2 words: {over[:3]}"
print("\nALL 3 TOKENIZERS OK")
