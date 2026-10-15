"""Generate the local procedural visual-search catalogue."""

import argparse
from pathlib import Path

from src.synthetic_data import DEFAULT_NUM_PRODUCTS, generate_synthetic_dataset


def compile_hardware_corpus(num_samples: int = DEFAULT_NUM_PRODUCTS * 10, seed: int = 42) -> dict:
    """Generate a dataset with the requested sample count (ten queries per product)."""
    if num_samples < 50 or num_samples % 50:
        raise ValueError("num_samples must be a multiple of 50 and at least 50.")

    return generate_synthetic_dataset(
        output_dir=Path(__file__).resolve().parent / "data" / "raw",
        seed=seed,
        num_products=num_samples // 10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_PRODUCTS * 10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = compile_hardware_corpus(num_samples=args.num_samples, seed=args.seed)
    print(
        f"Generated {summary['num_examples']} synthetic queries for "
        f"{summary['num_unique_products']} products "
        f"(seed={summary['seed']})."
    )


if __name__ == "__main__":
    main()
