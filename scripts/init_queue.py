#!/usr/bin/env python
"""
Initialize work queues for relayer scan experiments.

Creates a work queue file with all config indices to be processed.
Optionally skips configs that have already been completed.

Usage:
    # Initialize with all 1177 configs
    python scripts/init_queue.py --num-layers 48

    # Initialize with specific range
    python scripts/init_queue.py --num-layers 48 --config-start 0 --config-end 500

    # Skip configs from existing results
    python scripts/init_queue.py --num-layers 48 --skip-existing results/existing.pkl

    # Use strategic subset instead of full configs
    python scripts/init_queue.py --num-layers 48 --strategic

    # Limit span length (e.g., only duplicates with span <= 20)
    python scripts/init_queue.py --num-layers 92 --max-span 20

    # Sweep span ranges (e.g., 21-30)
    python scripts/init_queue.py --num-layers 92 --min-span 21 --max-span 30 --exclude-baseline
"""

import argparse
import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.layer_config import (
    layer_key,
    layer_spec_string,
    legacy_key_to_ij,
    normalize_to_layers,
    validate_layers,
)

def generate_layer_dict(num_layers: int) -> dict[tuple[int, int], list[int]]:
    """Generate the full (i, j) layer-dup config space plus baseline (0,0)."""
    layers_dict = {(0, 0): list(range(num_layers))}
    for j in range(num_layers + 1):
        for i in range(j):
            layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))
    return layers_dict


def generate_layer_dict_strategic(num_layers: int) -> dict[tuple[int, int], list[int]]:
    """
    Generate a strategic subset of configs.

    This mirrors the current strategic sampling used in worker tooling but keeps
    queue initialization independent from torch/transformers imports.
    """
    layers_dict = {(0, 0): list(range(num_layers))}

    early_end = num_layers // 4
    for i in range(early_end):
        j = i + 1
        layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))

    middle_start = num_layers * 3 // 8
    middle_end = num_layers * 5 // 8
    for i in range(middle_start, middle_end):
        j = i + 1
        layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))

    late_start = num_layers * 3 // 4
    for i in range(late_start, num_layers):
        j = min(i + 1, num_layers)
        layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))

    for start in range(0, num_layers, 5):
        end = min(start + 3, num_layers)
        layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    for start in [
        num_layers // 5,
        num_layers * 2 // 5,
        num_layers * 3 // 5,
        num_layers * 5 // 6,
        num_layers * 7 // 8,
    ]:
        end = min(start + 5, num_layers)
        layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    for start in range(0, num_layers, 8):
        end = min(start + 8, num_layers)
        layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    for start in range(0, num_layers, 12):
        end = min(start + 12, num_layers)
        layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    fib_points = [0, 1, 2, 4, 8, 16, 24, 32, 40, 44, num_layers]
    for idx in range(len(fib_points) - 1):
        start = fib_points[idx]
        end = min(fib_points[idx + 1], num_layers)
        if start < end:
            layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    return layers_dict


def main():
    parser = argparse.ArgumentParser(description="Initialize work queue for relayer experiments")
    parser.add_argument("--num-layers", type=int, default=48,
                        help="Number of layers in the base model")
    parser.add_argument("--strategic", action="store_true",
                        help="Use strategic subset (~92 configs) instead of full configs")
    parser.add_argument("--min-span", type=int, default=None,
                        help="Minimum duplicate span length (j - i).")
    parser.add_argument("--max-span", type=int, default=None,
                        help="Maximum duplicate span length (j - i).")
    parser.add_argument("--min-i", type=int, default=None,
                        help="Minimum block start i to include for generated legacy sweeps.")
    parser.add_argument("--max-i", type=int, default=None,
                        help="Maximum block start i to include for generated legacy sweeps.")
    parser.add_argument("--i-stride", type=int, default=1,
                        help="Keep only generated configs where i is on this stride.")
    parser.add_argument("--j-stride", type=int, default=1,
                        help="Keep only generated configs where j is on this stride.")
    parser.add_argument("--exclude-baseline", action="store_true",
                        help="Exclude baseline (0,0) from the queue.")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=(
            "Optional explicit config file. Each non-comment line can be "
            "`layers:...`, `blocks:...`, or legacy `(i,j)` / `i,j`."
        ),
    )
    parser.add_argument("--config-start", type=int, default=0,
                        help="Starting config index (inclusive)")
    parser.add_argument("--config-end", type=int, default=None,
                        help="Ending config index (exclusive)")
    parser.add_argument("--queue-file", type=str, default="work_queue.json",
                        help="Path to queue file (default: work_queue.json)")
    parser.add_argument("--results-file", type=str, default="results/shared_results.pkl",
                        help="Path to shared results file")
    parser.add_argument("--skip-existing", type=str, nargs="*", default=[],
                        help="Skip configs that exist in these results files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be initialized without writing")

    args = parser.parse_args()

    all_configs: list[dict] = []
    layers_dict: dict[tuple[int, int], list[int]] = {}

    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                layers = normalize_to_layers(args.num_layers, line)
                legacy_ij = legacy_key_to_ij(line)
                all_configs.append(
                    {
                        "idx": len(all_configs),
                        "layers": layers,
                        "spec": line if ":" in line else layer_spec_string(layers),
                        "legacy_key": legacy_ij,
                        "source_line": line_no,
                    }
                )
        print(f"Using explicit config file: {config_path} ({len(all_configs)} configs)")
    else:
        # Generate legacy sweep configs and convert to canonical layers.
        if args.strategic:
            layers_dict = generate_layer_dict_strategic(args.num_layers)
            print(f"Using strategic subset: {len(layers_dict)} configs")
        else:
            layers_dict = generate_layer_dict(args.num_layers)
            print(f"Using full config space: {len(layers_dict)} configs")

        if args.i_stride < 1 or args.j_stride < 1:
            raise ValueError("--i-stride and --j-stride must be >= 1")

        if (
            args.min_i is not None
            or args.max_i is not None
            or args.i_stride != 1
            or args.j_stride != 1
        ):
            def index_ok(key):
                if key == (0, 0):
                    return not args.exclude_baseline
                i, j = key
                if args.min_i is not None and i < args.min_i:
                    return False
                if args.max_i is not None and i > args.max_i:
                    return False
                i_origin = args.min_i if args.min_i is not None else 0
                if (i - i_origin) % args.i_stride != 0:
                    return False
                if j % args.j_stride != 0:
                    return False
                return True

            layers_dict = {k: v for k, v in layers_dict.items() if index_ok(k)}
            print(f"After index/stride filter: {len(layers_dict)} configs")

        # Apply optional span filtering
        if args.min_span is not None or args.max_span is not None:
            def span_ok(key):
                if key == (0, 0):
                    return not args.exclude_baseline
                span = key[1] - key[0]
                if args.min_span is not None and span < args.min_span:
                    return False
                if args.max_span is not None and span > args.max_span:
                    return False
                return True

            layers_dict = {k: v for k, v in layers_dict.items() if span_ok(k)}
            print(f"After span filter: {len(layers_dict)} configs")

        if args.exclude_baseline and (0, 0) in layers_dict:
            layers_dict = {k: v for k, v in layers_dict.items() if k != (0, 0)}
            print(f"After baseline exclusion: {len(layers_dict)} configs")

        all_config_keys = list(layers_dict.keys())
        for idx, key in enumerate(all_config_keys):
            layers = layers_dict[key]
            all_configs.append(
                {
                    "idx": idx,
                    "layers": layers,
                    "spec": layer_spec_string(layers),
                    "legacy_key": key,
                    "source_line": None,
                }
            )

    total_configs = len(all_configs)
    config_end = args.config_end if args.config_end is not None else total_configs
    selected_configs = all_configs[args.config_start:config_end]

    print(f"Config range: [{args.config_start}:{config_end}] = {len(selected_configs)} configs")

    # Collect completed configs from existing results files (canonical layer keys)
    completed_keys: set[tuple[int, ...]] = set()

    def _normalize_existing_key(raw_key) -> tuple[int, ...] | None:
        legacy_ij = legacy_key_to_ij(raw_key)
        if legacy_ij is not None and legacy_ij in layers_dict:
            return layer_key(layers_dict[legacy_ij])
        try:
            layers = normalize_to_layers(args.num_layers, raw_key)
        except Exception:
            return None
        try:
            validate_layers(args.num_layers, layers)
        except Exception:
            return None
        return layer_key(layers)

    # Check main results file
    results_path = Path(args.results_file)
    if results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
            for raw_key in results.keys():
                normalized = _normalize_existing_key(raw_key)
                if normalized is not None:
                    completed_keys.add(normalized)
        print(f"Found {len(results)} raw completed entries in {results_path}")

    # Check additional results files
    for rf in args.skip_existing:
        rf_path = Path(rf)
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                results = pickle.load(f)
                for raw_key in results.keys():
                    normalized = _normalize_existing_key(raw_key)
                    if normalized is not None:
                        completed_keys.add(normalized)
            print(f"Found {len(results)} raw completed entries in {rf_path}")

    # Filter out completed configs
    pending_configs = [
        cfg for cfg in selected_configs if layer_key(cfg["layers"]) not in completed_keys
    ]

    print("\nSummary:")
    print(f"  Total configs in range: {len(selected_configs)}")
    print(f"  Already completed: {len(selected_configs) - len(pending_configs)}")
    print(f"  Pending to process: {len(pending_configs)}")

    if args.dry_run:
        print("\n[DRY RUN] Would initialize queue with:")
        print(f"  Queue file: {args.queue_file}")
        print(f"  Results file: {args.results_file}")
        sample_first = [cfg["spec"] for cfg in pending_configs[:5]]
        sample_last = [cfg["spec"] for cfg in pending_configs[-5:]]
        print(f"  First 5 pending configs: {sample_first}")
        print(f"  Last 5 pending configs: {sample_last}")
        return

    # Build canonical queue entries with explicit layers.
    queue_entries = []
    for cfg in pending_configs:
        entry = {
            "idx": int(cfg["idx"]),
            "layers": list(cfg["layers"]),
            "spec": str(cfg["spec"]),
        }
        legacy_key = cfg.get("legacy_key")
        if isinstance(legacy_key, tuple) and len(legacy_key) == 2:
            entry["key"] = [int(legacy_key[0]), int(legacy_key[1])]
        queue_entries.append(entry)

    # Create queue file
    queue_path = Path(args.queue_file)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    with open(queue_path, 'w') as f:
        json.dump(queue_entries, f)

    print(f"\nCreated queue file: {queue_path}")

    # Initialize results file if it doesn't exist
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        with open(results_path, 'wb') as f:
            pickle.dump({}, f)
        print(f"Created empty results file: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Work queue initialized successfully!")
    print(f"{'='*60}")
    print(f"  Queue file: {queue_path}")
    print(f"  Results file: {results_path}")
    print(f"  Configs to process: {len(queue_entries)}")
    print("\nLaunch examples (choose worker type):")
    print(
        "  CUDA_VISIBLE_DEVICES=0 uv run python -m src.workers.math_worker "
        f"--model-path /path/to/model --queue-file {args.queue_file} "
        f"--results-file {args.results_file} --dataset-path datasets/math_16.json --batch-size 16 &"
    )
    print(
        "  CUDA_VISIBLE_DEVICES=0 uv run python -m src.workers.eq_worker "
        f"--model-path /path/to/model --queue-file {args.queue_file} "
        f"--results-file {args.results_file} --dataset-path datasets/eq_16.json --batch-size 5 &"
    )


if __name__ == "__main__":
    main()
