"""Local sanity checks for both BPE methods. Not part of submission."""
import tempfile, os
from bpe_tokenizer import BPETokenizer, WORD_MARKER
from base_tokenizer import BaseTokenizer

corpus = [
    "New York is in the United States\n",
    "I love New York in the summer\n",
    "the cat sat on the mat in the house\n",
    "of the people by the people for the people\n",
    "New York New York the the the\n",
    "Hello great job 123 zzqx! is mat\n",
] * 100

samples = [
    "New York is great",
    "the cat sat on the mat",
    "of the people",
    "Hello world!",
    "completely unseen zzqx 123",
]


def check(method):
    print(f"\n========== method={method} ==========")
    tok = BPETokenizer(vocab_size=300, method=method, num_bigrams=5, max_train_chars=10_000_000)
    tok.train(corpus)
    print("vocab:", tok.get_vocab_size(), "| space_token:", repr(tok.space_token),
          "| in vocab:", tok.space_token in tok.token_to_id)
    assert tok.space_token and tok.space_token in tok.token_to_id

    specials = set(tok.special_tokens)
    # bigram present + NO token exceeds 2 words (<=1 internal space)
    bigrams = []
    for t in tok.token_to_id:
        if t in specials:
            continue
        surface = t.replace(tok.space_token, " ") if tok.space_token else t
        n_internal = surface.strip().count(" ")
        assert n_internal <= 1, f">2-word token: {t!r} ({n_internal} internal spaces)"
        if n_internal == 1:
            bigrams.append(t)
    print("bigram tokens:", len(bigrams), "| e.g.", [repr(b) for b in bigrams[:6]])
    assert bigrams, "no bigram -> disqualified"
    print("top_bigrams:", tok.top_bigrams[:3], "| least:", tok.least_bigrams[:2])

    for s in samples:
        ids = tok.encode(s)
        back = tok.decode(ids)
        ok = back.replace(" ", "") == s.replace(" ", "")
        print(f"  recon={ok} | {s!r} -> {len(ids)} ids -> {back!r}")
        assert ok, f"reconstruction failed for {s!r}"
        assert all(isinstance(i, int) for i in ids)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.pkl")
        tok.save(p)
        loaded = BaseTokenizer.load(p)
        assert loaded.encode("New York is great") == tok.encode("New York is great")
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    print(f"method={method}: OK")


check("word")
check("byte")
print("\nALL SANITY CHECKS PASSED")
