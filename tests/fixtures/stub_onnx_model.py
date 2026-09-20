"""Builds a tiny *synthetic* fastembed-compatible model directory for offline runtime tests.

This is **not** the Jina model and says nothing about embedding quality. It exists to exercise the
real fastembed + onnxruntime + tokenizers code path (loading, tokenizer truncation/padding, mean
pooling, normalisation, batching, tokenizer cloning) without any network access:

* ``onnx/model.onnx``: one ``Gather`` from a fixed pseudo-random ``(vocab, 768)`` table, output
  shaped ``(batch, seq, 768)`` like a transformer's last hidden state.
* a character-level tokenizer (every character is one token, plus ``[CLS]``/``[SEP]``), so token
  counts are exactly ``len(text) + 2`` and truncation is easy to reason about.
* config files with ``model_max_length`` = ``MODEL_MAX_LENGTH``.

It is meant to be loaded through fastembed's ``specific_model_path`` argument under the registry
name of a real supported model (the tests use the Jina code model's 768-dimensional entry).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_MAX_LENGTH = 32
DIMENSION = 768
_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
_CHARS = [chr(c) for c in range(32, 127)]


def build_stub_model_dir(target: Path) -> Path:
    """Write the stub model files into ``target`` and return it."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

    (target / "onnx").mkdir(parents=True, exist_ok=True)

    vocab = {tok: i for i, tok in enumerate(_SPECIALS + _CHARS)}
    table = np.random.default_rng(1234).standard_normal((len(vocab), DIMENSION)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"], axis=0)],
        name="stub",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.FLOAT, ["batch", "seq", DIMENSION]
            )
        ],
        initializer=[numpy_helper.from_array(table, name="table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(target / "onnx" / "model.onnx"))

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer.save(str(target / "tokenizer.json"))

    (target / "config.json").write_text(
        json.dumps({"pad_token_id": 0, "max_position_embeddings": MODEL_MAX_LENGTH}), "utf-8"
    )
    (target / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": MODEL_MAX_LENGTH, "pad_token": "[PAD]"}), "utf-8"
    )
    (target / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            }
        ),
        "utf-8",
    )
    return target
