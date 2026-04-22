from __future__ import annotations

import argparse
import gc
import json
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from hf_export.common import (
    build_export_spec,
    build_exported_config,
    build_tensor_name_mapping,
    collect_layer_tensors,
    duplication_counts,
    load_json,
    save_json,
)


MODEL_INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "rys_export_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a relayered Hugging Face model by physically duplicating decoder layers in safetensors format.",
    )
    parser.add_argument("--source", required=True, type=Path, help="Path to the base HF model directory.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write the exported model into.")
    parser.add_argument(
        "--source-repo-id",
        default=None,
        help="Optional source HF repo id, recorded in the export manifest.",
    )
    parser.add_argument("--spec", default=None, help="Canonical config spec, e.g. 'blocks:30,34' or 'layers:0,1,2,2,3'.")
    parser.add_argument("--blocks", default=None, help="Block shorthand, e.g. '30,34;43,45'.")
    parser.add_argument("--layer-list", default=None, help="Explicit layer list, e.g. '0,1,2,3,3,4'.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and emit manifests/config only; do not write safetensors shards.")
    return parser.parse_args()


def copy_static_files(source_dir: Path, output_dir: Path, *, skip_files: set[str] | None = None) -> None:
    skip_files = skip_files or set()
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name == MODEL_INDEX_NAME:
            continue
        if item.name in skip_files:
            continue
        if item.name.startswith("model.safetensors-") and item.name.endswith(".safetensors"):
            continue
        if item.name == ".cache":
            continue
        target = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def build_output_weight_map(
    *,
    source_weight_map: dict[str, str],
    name_mapping: dict[str, str],
    shard_names: list[str],
    layer_indices: tuple[int, ...],
    text_layer_prefix: str,
) -> dict[str, str]:
    output_weight_map: dict[str, str] = {}
    layer_assignment = {
        new_pos: shard_names[min((new_pos * len(shard_names)) // max(len(layer_indices), 1), len(shard_names) - 1)]
        for new_pos in range(len(layer_indices))
    }

    for new_key, old_key in name_mapping.items():
        if new_key.startswith(text_layer_prefix):
            layer_idx = int(new_key[len(text_layer_prefix):].split(".", 1)[0])
            output_weight_map[new_key] = layer_assignment[layer_idx]
        else:
            output_weight_map[new_key] = source_weight_map[old_key]
    return output_weight_map


def write_shards(
    *,
    source_dir: Path,
    output_dir: Path,
    output_weight_map: dict[str, str],
    name_mapping: dict[str, str],
    source_weight_map: dict[str, str],
) -> None:
    # Group new tensors by output shard
    grouped: dict[str, list[tuple[str, str]]] = {}
    for new_key, target_file in output_weight_map.items():
        grouped.setdefault(target_file, []).append((new_key, name_mapping[new_key]))

    for shard_name in sorted(grouped):
        entries = grouped[shard_name]
        
        # Figure out which source files this shard needs
        needed_sources = set()
        for _, old_key in entries:
            needed_sources.add(source_weight_map[old_key])
        
        # Open only those source files
        tensors = {}
        with ExitStack() as stack:
            handles = {
                sf: stack.enter_context(safe_open(source_dir / sf, framework="pt", device="cpu"))
                for sf in needed_sources
            }
            for new_key, old_key in entries:
                source_file = source_weight_map[old_key]
                tensors[new_key] = handles[source_file].get_tensor(old_key).clone()
        
        # Handles are closed here before we write
        target_path = output_dir / shard_name
        print(f"[export] writing {target_path.name} with {len(entries)} tensors")
        save_file(tensors, str(target_path))
        del tensors
        gc.collect()


def build_manifest(
    *,
    source_dir: Path,
    source_repo_id: str | None,
    output_dir: Path,
    spec_text: str,
    layer_indices: tuple[int, ...],
    source_num_layers: int,
    text_layer_prefix: str,
) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "source_repo_id": source_repo_id,
        "output_dir": str(output_dir),
        "source_num_layers": source_num_layers,
        "target_num_layers": len(layer_indices),
        "extra_layers": len(layer_indices) - source_num_layers,
        "text_layer_prefix": text_layer_prefix,
        "spec": spec_text,
        "layer_indices": list(layer_indices),
        "duplication_counts": duplication_counts(layer_indices),
    }


def main() -> None:
    args = parse_args()
    spec = build_export_spec(
        source_dir=args.source.resolve(),
        output_dir=args.output.resolve(),
        source_repo_id=args.source_repo_id,
        spec=args.spec,
        blocks=args.blocks,
        layer_list=args.layer_list,
    )

    if spec.output_dir == spec.source_dir or spec.source_dir in spec.output_dir.parents:
        raise SystemExit("Output directory must not be inside the source model directory.")

    if spec.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {spec.output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(spec.output_dir)

    spec.output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_json(spec.source_dir / "config.json")
    index_json = load_json(spec.source_dir / MODEL_INDEX_NAME)
    source_weight_map = index_json["weight_map"]
    name_mapping = build_tensor_name_mapping(
        weight_map=source_weight_map,
        text_layer_prefix=spec.text_layer_prefix,
        layer_indices=spec.layer_indices,
    )
    tensors_by_layer = collect_layer_tensors(source_weight_map, spec.text_layer_prefix)

    if spec.source_num_layers != len(tensors_by_layer):
        raise SystemExit(
            f"Config says {spec.source_num_layers} layers, but index exposes {len(tensors_by_layer)} layers for "
            f"{spec.text_layer_prefix}. Refusing to export."
        )

    shard_names = sorted(set(source_weight_map.values()))
    output_weight_map = build_output_weight_map(
        source_weight_map=source_weight_map,
        name_mapping=name_mapping,
        shard_names=shard_names,
        layer_indices=spec.layer_indices,
        text_layer_prefix=spec.text_layer_prefix,
    )

    exported_config = build_exported_config(
        base_config,
        layer_indices=spec.layer_indices,
        source_num_layers=spec.source_num_layers,
        source_repo_id=spec.source_repo_id,
        spec_text=spec.spec_text,
        text_layer_prefix=spec.text_layer_prefix,
    )
    copy_static_files(spec.source_dir, spec.output_dir, skip_files=set(source_weight_map.values()))
    save_json(spec.output_dir / "config.json", exported_config)

    manifest = build_manifest(
        source_dir=spec.source_dir,
        source_repo_id=spec.source_repo_id,
        output_dir=spec.output_dir,
        spec_text=spec.spec_text,
        layer_indices=spec.layer_indices,
        source_num_layers=spec.source_num_layers,
        text_layer_prefix=spec.text_layer_prefix,
    )
    save_json(spec.output_dir / MANIFEST_NAME, manifest)

    index_payload = {
        "metadata": {"total_size": 0},
        "weight_map": output_weight_map,
    }
    save_json(spec.output_dir / MODEL_INDEX_NAME, index_payload)

    print(f"[export] source layers={spec.source_num_layers}, target layers={len(spec.layer_indices)}")
    print(f"[export] spec={spec.spec_text}")
    print(f"[export] duplicated layers={json.dumps(duplication_counts(spec.layer_indices), sort_keys=True)}")
    print(f"[export] output={spec.output_dir}")

    if args.dry_run:
        print("[export] dry-run only; shards were not written.")
        return

    write_shards(
        source_dir=spec.source_dir,
        output_dir=spec.output_dir,
        output_weight_map=output_weight_map,
        name_mapping=name_mapping,
        source_weight_map=source_weight_map,
    )

    total_size = sum(path.stat().st_size for path in spec.output_dir.glob("model.safetensors-*.safetensors"))
    index_payload["metadata"]["total_size"] = int(total_size)
    save_json(spec.output_dir / MODEL_INDEX_NAME, index_payload)
    print(f"[export] wrote {len(shard_names)} shard files, total_size={total_size}")


if __name__ == "__main__":
    main()
