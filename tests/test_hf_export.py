from hf_export.common import (
    build_exported_config,
    build_tensor_name_mapping,
    detect_text_layer_prefix,
)
from hf_export.export_model import copy_static_files


def test_build_tensor_name_mapping_duplicates_decoder_layers() -> None:
    weight_map = {
        "model.layers.0.a": "model-00001.safetensors",
        "model.layers.0.b": "model-00001.safetensors",
        "model.layers.1.a": "model-00001.safetensors",
        "model.layers.1.b": "model-00001.safetensors",
        "model.layers.2.a": "model-00002.safetensors",
        "model.layers.2.b": "model-00002.safetensors",
        "model.norm.weight": "model-00002.safetensors",
    }
    mapping = build_tensor_name_mapping(
        weight_map=weight_map,
        text_layer_prefix="model.layers.",
        layer_indices=(0, 1, 1, 2),
    )
    assert mapping["model.layers.0.a"] == "model.layers.0.a"
    assert mapping["model.layers.2.a"] == "model.layers.1.a"
    assert mapping["model.layers.2.b"] == "model.layers.1.b"
    assert mapping["model.layers.3.a"] == "model.layers.2.a"
    assert mapping["model.norm.weight"] == "model.norm.weight"


def test_detect_text_layer_prefix_supports_kimi_k25_inner_language_model() -> None:
    weight_map = {
        "language_model.model.layers.0.self_attn.kv_b_proj.weight": "model-00001.safetensors",
        "language_model.model.layers.1.mlp.experts.0.up_proj.weight_packed": "model-00002.safetensors",
        "vision_tower.blocks.0.attn.qkv.weight": "model-00003.safetensors",
    }
    assert detect_text_layer_prefix(weight_map) == "language_model.model.layers."


def test_build_tensor_name_mapping_duplicates_kimi_compressed_tensors() -> None:
    weight_map = {
        "language_model.model.layers.0.self_attn.kv_b_proj.weight": "model-00001.safetensors",
        "language_model.model.layers.1.mlp.experts.0.up_proj.weight_shape": "model-00002.safetensors",
        "language_model.model.layers.1.mlp.experts.0.up_proj.weight_scale": "model-00002.safetensors",
        "language_model.model.layers.1.mlp.experts.0.up_proj.weight_packed": "model-00002.safetensors",
        "language_model.model.norm.weight": "model-00003.safetensors",
    }
    mapping = build_tensor_name_mapping(
        weight_map=weight_map,
        text_layer_prefix="language_model.model.layers.",
        layer_indices=(0, 1, 1),
    )
    assert (
        mapping["language_model.model.layers.2.mlp.experts.0.up_proj.weight_shape"]
        == "language_model.model.layers.1.mlp.experts.0.up_proj.weight_shape"
    )
    assert (
        mapping["language_model.model.layers.2.mlp.experts.0.up_proj.weight_scale"]
        == "language_model.model.layers.1.mlp.experts.0.up_proj.weight_scale"
    )
    assert (
        mapping["language_model.model.layers.2.mlp.experts.0.up_proj.weight_packed"]
        == "language_model.model.layers.1.mlp.experts.0.up_proj.weight_packed"
    )
    assert mapping["language_model.model.norm.weight"] == "language_model.model.norm.weight"


def test_copy_static_files_skips_index_weight_map_shards(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text("{}")
    (source / "model-00001-of-00064.safetensors").write_text("large")
    (source / "model.safetensors-00001-of-00002.safetensors").write_text("legacy")

    copy_static_files(
        source,
        output,
        skip_files={"model-00001-of-00064.safetensors"},
    )

    assert (output / "config.json").exists()
    assert not (output / "model.safetensors.index.json").exists()
    assert not (output / "model-00001-of-00064.safetensors").exists()
    assert not (output / "model.safetensors-00001-of-00002.safetensors").exists()


def test_build_exported_config_updates_text_config_metadata() -> None:
    base_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["a", "b", "c", "d"],
        },
    }
    exported = build_exported_config(
        base_config,
        layer_indices=(0, 1, 1, 2, 3),
        source_num_layers=4,
        source_repo_id="example/base-model",
        spec_text="blocks:1,2",
        text_layer_prefix="model.layers.",
    )
    assert exported["text_config"]["num_hidden_layers"] == 5
    assert exported["text_config"]["layer_types"] == ["a", "b", "b", "c", "d"]
    assert exported["rys_relayer"]["layer_indices"] == [0, 1, 1, 2, 3]


def test_build_exported_config_remaps_modules_to_not_convert() -> None:
    base_config = {
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        },
        "quantization_config": {
            "modules_to_not_convert": [
                "lm_head",
                "model.layers.0.linear_attn.in_proj_a",
                "model.layers.1.linear_attn.in_proj_a",
                "model.layers.3.linear_attn.in_proj_b",
            ]
        },
    }
    exported = build_exported_config(
        base_config,
        layer_indices=(0, 1, 1, 2, 3),
        source_num_layers=4,
        source_repo_id="example/base-model",
        spec_text="layers:0,1,1,2,3",
        text_layer_prefix="model.layers.",
    )
    assert exported["quantization_config"]["modules_to_not_convert"] == [
        "lm_head",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.1.linear_attn.in_proj_a",
        "model.layers.2.linear_attn.in_proj_a",
        "model.layers.4.linear_attn.in_proj_b",
    ]
