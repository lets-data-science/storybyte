"""
02: Train a byte-level BPE tokenizer on TinyStories.

Byte-level BPE = the GPT-2 tokenization scheme (Radford et al. 2019): start from the
256 possible bytes, then repeatedly merge the most frequent adjacent pair. The trained
tokenizer is the ordered list of merges + the vocabulary.

We use the Hugging Face `tokenizers` (Rust) trainer for speed, then save it. The course
re-implements this exact algorithm from scratch in Module 1.

Output: data/tokenizer.json (+ vocab.json, merges.txt)
"""
import argparse, os
from tokenizers import ByteLevelBPETokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--min_frequency", type=int, default=2)
    a = ap.parse_args()

    train_txt = os.path.join(DATA, "train.txt")
    print(f"training byte-level BPE (vocab={a.vocab_size}) on {train_txt} ...")
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[train_txt],
        vocab_size=a.vocab_size,
        min_frequency=a.min_frequency,
        special_tokens=["<|endoftext|>"],
    )
    tok.save_model(DATA)                       # vocab.json + merges.txt
    tok.save(os.path.join(DATA, "tokenizer.json"))
    # sanity round-trip
    s = "Once upon a time, there was a little"
    enc = tok.encode(s)
    print(f"vocab size: {tok.get_vocab_size()}")
    print(f"encode({s!r}) -> {enc.ids}")
    print(f"  tokens: {enc.tokens}")
    print(f"decode round-trip ok: {tok.decode(enc.ids) == s}")
    print(f"saved tokenizer.json to {DATA}")
