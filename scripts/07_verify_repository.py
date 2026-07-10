#!/usr/bin/env python3
"""Fast, offline integrity checks for the checked-in StoryByte artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "course_artifacts"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

from reference_forward import StoryByteNumPy  # noqa: E402


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    required = [
        ART / "storybyte_config.json",
        ART / "storybyte_weights.npz",
        ART / "storybyte_tokenizer.json",
        ART / "storybyte_tokenizer_hf.json",
        ART / "train_traces.json",
        ART / "verification.json",
        ART / "interp_data.json",
        ART / "sample_generations.json",
    ]
    for path in required:
        check(path.is_file(), f"missing {path.relative_to(ROOT)}", failures)
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        return 1

    config = json.loads((ART / "storybyte_config.json").read_text())
    traces = json.loads((ART / "train_traces.json").read_text())
    verification = json.loads((ART / "verification.json").read_text())
    tokenizer = json.loads((ART / "storybyte_tokenizer.json").read_text())
    arrays = np.load(ART / "storybyte_weights.npz")

    check(config["n_params"] == 1_088_256, "config parameter total changed", failures)
    check(len(arrays.files) == 52, f"expected 52 arrays, found {len(arrays.files)}", failures)
    check(sum(arrays[key].size for key in arrays.files) == config["n_params"], "weight element total does not match config", failures)
    check({str(arrays[key].dtype) for key in arrays.files} == {"float32"}, "weights are not uniformly float32", failures)
    check(arrays["wte"].shape == (2048, 128), "unexpected token embedding shape", failures)
    check(arrays["wpe"].shape == (256, 128), "unexpected position embedding shape", failures)

    check(len(tokenizer["vocab"]) == 2048, "tokenizer vocabulary is not 2,048", failures)
    check(len(tokenizer["merges"]) == 1791, "tokenizer merge count is not 1,791", failures)
    check(tokenizer["special_tokens"].get("<|endoftext|>") == 0, "end-of-text ID is not 0", failures)

    check(config["final_step"] == traces["steps"][-1], "final trace step drift", failures)
    check(abs(config["final_train_loss"] - traces["train_loss"][-1]) < 5e-5, "final train loss drift", failures)
    check(abs(config["final_val_loss"] - traces["val_loss"][-1]) < 5e-5, "final validation loss drift", failures)
    check(abs(config["val_perplexity"] - math.exp(traces["val_loss"][-1])) < 5e-4, "final perplexity drift", failures)

    best_index = int(np.argmin(traces["val_loss"]))
    check(config["checkpoint_step"] == traces["steps"][best_index], "selected checkpoint step is not the trace minimum", failures)
    check(abs(config["checkpoint_val_loss"] - traces["val_loss"][best_index]) < 5e-5, "checkpoint validation loss drift", failures)
    check(abs(config["checkpoint_val_perplexity"] - math.exp(traces["val_loss"][best_index])) < 5e-4, "checkpoint perplexity drift", failures)

    check(verification.get("pass") is True, "recorded PyTorch/NumPy verification did not pass", failures)
    check(verification.get("dtype") == "float32", "verification dtype drift", failures)
    check(verification.get("numpy_vs_torch_max_logit_diff", 1) < 1e-3, "recorded maximum logit difference is too large", failures)
    check(verification.get("greedy_agreement") == 1.0, "recorded greedy agreement is not 100%", failures)

    model = StoryByteNumPy(str(ART))
    logits = model.forward(list(range(5, 25)))
    check(logits.shape == (20, 2048), f"unexpected forward shape {logits.shape}", failures)
    check(bool(np.isfinite(logits).all()), "forward pass produced non-finite logits", failures)

    scripts_copy = ROOT / "scripts" / "reference_forward.py"
    artifact_copy = ART / "reference_forward.py"
    check(scripts_copy.read_bytes() == artifact_copy.read_bytes(), "reference_forward.py copies have drifted", failures)

    if (DATA / "meta.json").is_file():
        meta = json.loads((DATA / "meta.json").read_text())
        check(meta["train_tokens"] == config["train_tokens"], "training token count drift", failures)
        check(meta["vocab_size"] == config["vocab_size"], "data/config vocabulary drift", failures)

    report = {
        "parameters": config["n_params"],
        "arrays": len(arrays.files),
        "dtype": config["dtype_export"],
        "finalTrace": {
            "step": config["final_step"],
            "validationLoss": config["final_val_loss"],
            "perplexity": config["val_perplexity"],
        },
        "selectedCheckpoint": {
            "step": config["checkpoint_step"],
            "validationLoss": config["checkpoint_val_loss"],
            "perplexity": config["checkpoint_val_perplexity"],
        },
        "forwardShape": list(logits.shape),
        "sha256": {
            "config": sha256(ART / "storybyte_config.json"),
            "weights": sha256(ART / "storybyte_weights.npz"),
            "tokenizer": sha256(ART / "storybyte_tokenizer.json"),
        },
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
