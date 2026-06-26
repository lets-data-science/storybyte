"""
01 — Download the TinyStories (V2, GPT-4) dataset.

TinyStories: Eldan & Li, Microsoft Research, 2023 (arXiv:2305.07759).
Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Full reproducible default = download the complete train + valid txt files.
For a fast local run, pass --subset_mb N to grab only the first N MB of the
(huge) train file via an HTTP range request (sufficient to train a ~1M-param model).
"""
import argparse, os, sys, urllib.request

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN = "TinyStoriesV2-GPT4-train.txt"
VALID = "TinyStoriesV2-GPT4-valid.txt"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def download(fname, dest, max_bytes=None):
    url = f"{BASE}/{fname}"
    headers = {"User-Agent": "storybyte/1.0"}
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    print(f"  GET {url}" + (f"  (first {max_bytes/1e6:.0f} MB)" if max_bytes else ""))
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); got += len(chunk)
            print(f"\r    {got/1e6:.0f} MB", end="", flush=True)
    print()
    # if we range-truncated mid-story, trim back to the last blank line for clean stories
    if max_bytes:
        with open(dest, "rb") as f:
            data = f.read()
        cut = data.rfind(b"\n\n")
        if cut > 0:
            with open(dest, "wb") as f:
                f.write(data[:cut])
    return os.path.getsize(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_mb", type=int, default=0, help="first N MB of train (0 = full file)")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    vsize = download(VALID, os.path.join(DATA, "valid.txt"))
    print(f"valid.txt: {vsize/1e6:.1f} MB")
    tsize = download(TRAIN, os.path.join(DATA, "train.txt"),
                     max_bytes=a.subset_mb * (1 << 20) if a.subset_mb else None)
    print(f"train.txt: {tsize/1e6:.1f} MB")
    print("done.")
