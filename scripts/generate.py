"""Sample text from a trained language-model checkpoint.

Usage::

    python scripts/generate.py --checkpoint runs/<run>/ckpt/last.pt --prompt "ROMEO:" \\
        --max-new-tokens 300 --temperature 0.8 --top-k 40

The tokenizer is loaded from ``tokenizer.json`` beside the checkpoint, so the
corpus is not needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import dlfp.models  # noqa: F401  (registers models)
from dlfp.config import from_dict
from dlfp.data.tokenizer import load_tokenizer
from dlfp.registry import build_model
from dlfp.training import load_checkpoint, restore_state
from dlfp.utils import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command-line flags.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="LM checkpoint (best.pt or last.pt).")
    parser.add_argument("--prompt", default="\n", help="Text to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: unseeded).")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    """Load model and tokenizer, sample, print the decoded text."""
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser()
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = from_dict(ckpt["config"])

    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"No tokenizer.json beside the checkpoint (looked for {tok_path}).")
    tokenizer = load_tokenizer(tok_path)

    model = build_model(cfg.model.name, **cfg.model.params)
    if getattr(model, "task", None) != "language_modeling" or not hasattr(model, "generate"):
        raise SystemExit(
            f"Checkpoint holds a '{cfg.model.name}' model with task={getattr(model, 'task', None)!r}; "
            f"generation needs a language model trained with model.params.mode: lm."
        )
    restore_state(ckpt, model=model, restore_rng=False)
    device = resolve_device(args.device)
    model.to(device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt) or [tokenizer.pad_idx], device=device)
    out = model.generate(prompt_ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
