#!/usr/bin/env python3
"""
One-shot tokenizer pipeline that **uses only adapters.py**.

Given:
  * a tokenizer training text file (single .txt),
  * a train text file (or directory of .txt),
  * a valid text file (or directory of .txt),

This script will:
  1) Train a BPE tokenizer with `run_train_bpe` on the tokenizer train file.
  2) Initialize a `Tokenizer(vocab, merges, special_tokens)` instance.
  3) Save the tokenizer to a directory (`vocab.json`, `merges.txt`, `meta.json`).
  4) Encode the train/valid text into token ids and save as `.npy` (1D int64),
     compatible with `train.py`.

No subcommands; just pass the arguments once.

Note: `run_train_bpe` expects a single text file path as input. If you have a
folder of many files to build the tokenizer, concatenate first (e.g.,
`cat data/tok/*.txt > tokenizer_corpus.txt`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List
import numpy as np

from tests.adapters import Tokenizer, run_train_bpe


# --------------------------- save / load helpers -----------------------------

def _save_tokenizer(out_dir: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save vocab.json as {token_str: id} -> refer to gpt2_vocab.json
    tok_to_id = {bytes_tok.decode("utf-8"): int(tid) for tid, bytes_tok in vocab.items()}
    (out / "vocab.json").write_text(json.dumps(tok_to_id, ensure_ascii=False))

    # Save merges.txt as "left right" per line -> refer to gpt2_merges.txt
    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.decode('utf-8')} {right.decode('utf-8')}\n")

    # Save meta.json with special tokens
    meta = {"special_tokens": special_tokens, "vocab_size": len(vocab)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))


# ------------------------------- encoding -----------------------------------

def _iter_txt_files(path: str) -> Iterable[Path]:
    p = Path(path)
    if p.is_dir():
        yield from sorted(p.rglob("*.txt"))
    else:
        yield p


def encode_corpus(tokenizer: Tokenizer, input_path: str) -> np.ndarray:
    ## Assume input files already have eos, we don't manually add it
    ids: List[int] = []
    for fp in _iter_txt_files(input_path):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ids.extend(tokenizer.encode(line))
    return np.asarray(ids)


# ----------------------------- one-shot pipeline -----------------------------

def fit_tokenizer_and_encode(
    tokenizer_train_file: str,
    vocab_size: int,
    tokenizer_out_dir: str,
    train_txt: str,
    valid_txt: str,
    special_tokens: List[str],
) -> None:
    ## Train tokenizer
    vocab, merges = run_train_bpe(tokenizer_train_file, vocab_size, special_tokens=special_tokens)
    tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    ## Save tokenizer
    _save_tokenizer(tokenizer_out_dir, vocab, merges, special_tokens=special_tokens)

    ## Encode train/valid and save them
    train_ids = encode_corpus(tok, train_txt)
    valid_ids = encode_corpus(tok, valid_txt)
    train_out_npy = train_txt.replace(".txt", ".npy")
    valid_out_npy = valid_txt.replace(".txt", ".npy")
    Path(train_out_npy).parent.mkdir(parents=True, exist_ok=True)
    Path(valid_out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(train_out_npy, train_ids)
    np.save(valid_out_npy, valid_ids)

    print(
        f"Tokenizer saved to {tokenizer_out_dir} | "
        f"train tokens: {train_ids.size} → {train_out_npy} | "
        f"valid tokens: {valid_ids.size} → {valid_out_npy}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer on one file, save it, then encode train/valid to .npy (adapters.py)")
    parser.add_argument("--tokenizer_train", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--tokenizer_out", type=str, default="save/tokenizer")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--train_txt", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_txt", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--special", action="append", default=["<|endoftext|>"], help="Special tokens")

    args = parser.parse_args()

    fit_tokenizer_and_encode(
        tokenizer_train_file=args.tokenizer_train,
        vocab_size=args.vocab_size,
        tokenizer_out_dir=args.tokenizer_out,
        train_txt=args.train_txt,
        valid_txt=args.valid_txt,
        special_tokens=args.special,
    )


if __name__ == "__main__":
    main()
