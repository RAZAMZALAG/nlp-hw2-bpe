import argparse
import os
from typing import List

from bpe_tokenizer import BPETokenizer


def read_text_file(file_path: str) -> List[str]:
    """
    Read lines from a text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of lines from the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def train_tokenizer(domain_file, output_dir: str, vocab_size: int = 5000,
                    output_name: str = "tokenizer.pkl"):
    """
    Train a tokenizer on domain data and save it.

    Args:
        domain_file: Path (or list of paths) to the domain training data. Multiple files are
            concatenated — used for tokenizer_3, whose corpus is domain_1 + domain_2.
        output_dir: Directory where to save the trained tokenizer
        vocab_size: Maximum vocabulary size
        output_name: Output filename (e.g. tokenizer_1.pkl) so each tokenizer can use its own vocab_size
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read domain data (accept a single path or several)
    paths = [domain_file] if isinstance(domain_file, str) else list(domain_file)
    texts = []
    for p in paths:
        print(f"Reading domain data from {p}")
        texts.extend(read_text_file(p))
    print(f"Read {len(texts)} lines of text from {len(paths)} file(s)")

    # Initialize and train tokenizer
    print(f"Training BPE tokenizer with vocab size {vocab_size}")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)

    # Save the tokenizer
    output_path = os.path.join(output_dir, output_name)
    print(f"Saving tokenizer to {output_path}")
    tokenizer.save(output_path)
    print(f"Tokenizer trained with {tokenizer.get_vocab_size()} tokens")
    
    # Test the tokenizer on a sample
    if texts:
        sample_text = texts[0].strip()
        print("\nExample encoding/decoding:")
        print(f"Original text: {sample_text}")
        
        encoded = tokenizer.encode(sample_text)
        print(f"Encoded: {encoded[:50]}{'...' if len(encoded) > 50 else ''}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on domain data")
    parser.add_argument("--domain_file", type=str, nargs="+", required=True,
                        help="Path(s) to the domain data file(s); multiple are concatenated (tokenizer_3)")
    parser.add_argument("--output_dir", type=str, default="tokenizers", help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Maximum vocabulary size")
    parser.add_argument("--output_name", type=str, default="tokenizer.pkl",
                        help="Output filename, e.g. tokenizer_1.pkl (lets each tokenizer use its own vocab_size)")

    args = parser.parse_args()

    train_tokenizer(args.domain_file, args.output_dir, args.vocab_size, args.output_name)