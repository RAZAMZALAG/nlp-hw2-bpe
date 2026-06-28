"""DEV-ONLY: cold encode speed + efficiency. Fresh load per run, single pass (grader-realistic).
Usage (from code_env_example/):  py code/_speedeval.py <tok.pkl> <dev.txt> [label]"""
import sys, time, pickle

def load_texts(p):
    with open(p, encoding="utf-8") as f:
        return [l.strip() for l in f]

def run(tok_path, dev_path, label=""):
    with open(tok_path, "rb") as f:
        tok = pickle.load(f)            # fresh instance -> cold cache
    texts = load_texts(dev_path)
    chars = sum(len(t) for t in texts)
    t0 = time.time()
    ntok = 0
    for t in texts:
        ntok += len(tok.encode(t))
    dt = time.time() - t0
    print(f"{label:>22} | cold {ntok/dt:>10.0f} tok/s | tok/char {ntok/chars:>7.4f} | "
          f"{ntok} tok / {chars} ch / {dt:.3f}s", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else sys.argv[1])
