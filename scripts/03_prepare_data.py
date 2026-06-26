"""
03 — Tokenize TinyStories and pack into flat uint16 token streams for training.

Each story is encoded with the trained BPE and joined by the <|endoftext|> token, then
the whole stream is written as a memory-mappable uint16 array (nanoGPT convention).

Output: data/train.bin, data/val.bin, data/meta.json
"""
import argparse, json, os
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def encode_file(tok, path, eot_id, limit_mb=0):
    ids = []
    nbytes = 0
    limit = limit_mb * (1 << 20) if limit_mb else None
    # TinyStories stories are separated by a blank line in the txt
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buf = []
        n_stories = 0
        for line in f:
            nbytes += len(line.encode("utf-8"))
            if line.strip() == "":
                if buf:
                    story = "".join(buf).strip()
                    ids.extend(tok.encode(story).ids)
                    ids.append(eot_id)
                    n_stories += 1
                    buf = []
                if limit and nbytes > limit:
                    break
            else:
                buf.append(line)
        if buf and not (limit and nbytes > limit):
            ids.extend(tok.encode("".join(buf).strip()).ids); ids.append(eot_id); n_stories += 1
    return np.array(ids, dtype=np.uint16), n_stories


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_mb", type=int, default=0, help="cap train text at N MB (0 = all)")
    a = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None, "missing <|endoftext|>"

    print("encoding train ...")
    train_ids, n_tr = encode_file(tok, os.path.join(DATA, "train.txt"), eot_id, a.train_mb)
    print("encoding val ...")
    val_ids, n_va = encode_file(tok, os.path.join(DATA, "valid.txt"), eot_id, 0)

    train_ids.tofile(os.path.join(DATA, "train.bin"))
    val_ids.tofile(os.path.join(DATA, "val.bin"))
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id,
        "train_tokens": int(train_ids.size), "val_tokens": int(val_ids.size),
        "train_stories": n_tr, "val_stories": n_va,
        "dtype": "uint16",
    }
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"train.bin = {train_ids.size:,} tokens ({n_tr:,} stories)")
    print(f"val.bin   = {val_ids.size:,} tokens ({n_va:,} stories)")
