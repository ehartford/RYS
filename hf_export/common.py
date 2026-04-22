from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.layer_config import normalize_to_layers


TEXT_LAYER_PREFIX_CANDIDATES = (
    "language_model.model.layers.",
    "model.language_model.layers.",
    "model.layers.",
    "language_model.layers.",
)


@dataclass(frozen=True)
class ExportSpec:
    source_dir: Path
    output_dir: Path
    layer_indices: tuple[int, ...]
    source_repo_id: str | None
    spec_text: str
    source_num_layers: int
    text_layer_prefix: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def parse_cli_spec(
    *,
    num_layers: int,
    spec: str | None,
    blocks: str | None,
    layer_list: str | None,
) -> tuple[tuple[int, ...], str]:
    selected = [(name, value) for name, value in (("spec", spec), ("blocks", blocks), ("layer-list", layer_list)) if value]
    if len(selected) != 1:
        raise ValueError("Specify exactly one of --spec, --blocks, or --layer-list.")

    name, raw_value = selected[0]
    if name == "blocks":
        spec_text = f"blocks:{raw_value}"
    elif name == "layer-list":
        spec_text = f"layers:{raw_value}"
    else:
        spec_text = raw_value

    layer_indices = tuple(normalize_to_layers(num_layers, spec_text))
    return layer_indices, spec_text


def detect_text_layer_prefix(weight_map: dict[str, str]) -> str:
    for prefix in TEXT_LAYER_PREFIX_CANDIDATES:
        if any(key.startswith(prefix) for key in weight_map):
            return prefix
    raise ValueError(
        "Could not detect decoder layer prefix in model.safetensors.index.json. "
        f"Tried: {', '.join(TEXT_LAYER_PREFIX_CANDIDATES)}"
    )


def collect_layer_tensors(weight_map: dict[str, str], text_layer_prefix: str) -> dict[int, dict[str, str]]:
    layer_re = re.compile(rf"^{re.escape(text_layer_prefix)}(\d+)(\..+)$")
    tensors_by_layer: dict[int, dict[str, str]] = {}
    for key in weight_map:
        match = layer_re.match(key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        suffix = match.group(2)
        tensors_by_layer.setdefault(layer_idx, {})[suffix] = key

    if not tensors_by_layer:
        raise ValueError(f"No tensors matched layer prefix {text_layer_prefix!r}.")
    return tensors_by_layer


def duplication_counts(layer_indices: tuple[int, ...]) -> dict[int, int]:
    return {
        int(layer_idx): int(count)
        for layer_idx, count in sorted(Counter(layer_indices).items())
        if count > 1
    }


def build_tensor_name_mapping(
    *,
    weight_map: dict[str, str],
    text_layer_prefix: str,
    layer_indices: tuple[int, ...],
) -> dict[str, str]:
    tensors_by_layer = collect_layer_tensors(weight_map, text_layer_prefix)
    expected_source_layers = set(range(max(tensors_by_layer) + 1))
    if set(tensors_by_layer) != expected_source_layers:
        missing = sorted(expected_source_layers - set(tensors_by_layer))
        raise ValueError(f"Layer tensor map is sparse. Missing layers: {missing}")

    mapping: dict[str, str] = {}
    for new_pos, old_pos in enumerate(layer_indices):
        for suffix, old_key in sorted(tensors_by_layer[old_pos].items()):
            mapping[f"{text_layer_prefix}{new_pos}{suffix}"] = old_key

    for key in weight_map:
        if not key.startswith(text_layer_prefix):
            mapping[key] = key

    return mapping


def build_exported_config(
    base_config: dict[str, Any],
    *,
    layer_indices: tuple[int, ...],
    source_num_layers: int,
    source_repo_id: str | None,
    spec_text: str,
    text_layer_prefix: str,
) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    target_num_layers = len(layer_indices)

    if cfg.get("num_hidden_layers") == source_num_layers:
        cfg["num_hidden_layers"] = target_num_layers

    if isinstance(cfg.get("layer_types"), list) and len(cfg["layer_types"]) == source_num_layers:
        cfg["layer_types"] = [cfg["layer_types"][idx] for idx in layer_indices]

    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        if text_cfg.get("num_hidden_layers") == source_num_layers:
            text_cfg["num_hidden_layers"] = target_num_layers
        if isinstance(text_cfg.get("layer_types"), list) and len(text_cfg["layer_types"]) == source_num_layers:
            text_cfg["layer_types"] = [text_cfg["layer_types"][idx] for idx in layer_indices]

    quant_cfg = cfg.get("quantization_config")
    if isinstance(quant_cfg, dict):
        modules_to_not_convert = quant_cfg.get("modules_to_not_convert")
        if isinstance(modules_to_not_convert, list) and all(isinstance(x, str) for x in modules_to_not_convert):
            positions_by_source: dict[int, list[int]] = {}
            for new_pos, old_pos in enumerate(layer_indices):
                positions_by_source.setdefault(int(old_pos), []).append(int(new_pos))

            layer_item_re = re.compile(rf"^{re.escape(text_layer_prefix)}(\d+)(\..+)$")
            remapped_modules: list[str] = []
            for item in modules_to_not_convert:
                match = layer_item_re.match(item)
                if not match:
                    remapped_modules.append(item)
                    continue
                old_idx = int(match.group(1))
                suffix = match.group(2)
                new_positions = positions_by_source.get(old_idx, [])
                for new_pos in new_positions:
                    remapped_modules.append(f"{text_layer_prefix}{new_pos}{suffix}")
            quant_cfg["modules_to_not_convert"] = remapped_modules

    cfg["rys_relayer"] = {
        "source_repo_id": source_repo_id,
        "source_num_layers": source_num_layers,
        "target_num_layers": target_num_layers,
        "spec": spec_text,
        "layer_indices": list(layer_indices),
        "duplication_counts": duplication_counts(layer_indices),
    }
    return cfg


def count_source_layers(base_config: dict[str, Any], tensors_by_layer: dict[int, dict[str, str]]) -> int:
    text_cfg = base_config.get("text_config")
    if isinstance(text_cfg, dict) and isinstance(text_cfg.get("num_hidden_layers"), int):
        return int(text_cfg["num_hidden_layers"])
    if isinstance(base_config.get("num_hidden_layers"), int):
        return int(base_config["num_hidden_layers"])
    return max(tensors_by_layer) + 1


def build_export_spec(
    *,
    source_dir: Path,
    output_dir: Path,
    source_repo_id: str | None,
    spec: str | None,
    blocks: str | None,
    layer_list: str | None,
) -> ExportSpec:
    base_config = load_json(source_dir / "config.json")
    index_json = load_json(source_dir / "model.safetensors.index.json")
    weight_map = index_json["weight_map"]
    text_layer_prefix = detect_text_layer_prefix(weight_map)
    tensors_by_layer = collect_layer_tensors(weight_map, text_layer_prefix)
    source_num_layers = count_source_layers(base_config, tensors_by_layer)
    layer_indices, spec_text = parse_cli_spec(
        num_layers=source_num_layers,
        spec=spec,
        blocks=blocks,
        layer_list=layer_list,
    )
    return ExportSpec(
        source_dir=source_dir,
        output_dir=output_dir,
        layer_indices=layer_indices,
        source_repo_id=source_repo_id,
        spec_text=spec_text,
        source_num_layers=source_num_layers,
        text_layer_prefix=text_layer_prefix,
    )
