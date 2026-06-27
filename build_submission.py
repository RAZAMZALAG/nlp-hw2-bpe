"""Build HW2_206922478.zip from submission/. Run after placing report_206922478.pdf in submission/."""
import os, sys, zipfile

REQUIRED = [
    "code/bpe_tokenizer.py",
    "trained_tokenizers/tokenizer_1.pkl",
    "trained_tokenizers/tokenizer_2.pkl",
    "trained_tokenizers/tokenizer_3.pkl",
    "report_206922478.pdf",
    "train_commands.txt",
]
missing = [f for f in REQUIRED if not os.path.exists(os.path.join("submission", f))]
if missing:
    print("MISSING (add these to submission/ first):")
    for m in missing:
        print("  -", m)
    sys.exit(1)

with zipfile.ZipFile("HW2_206922478.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk("submission"):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, "submission"))
print("Built HW2_206922478.zip")
print("Validate on the VM: uv run python check_submission.py HW2_206922478.zip")
