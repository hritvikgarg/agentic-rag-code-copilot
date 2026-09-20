"""FastEmbedEmbedder against a fake fastembed ``TextEmbedding`` (no model, no network)."""

import sys

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from copilot.embeddings import (
    EmbeddingConfigError,
    EmbeddingInputError,
    EmbeddingModelUnavailableError,
    EmbeddingRuntimeError,
    FastEmbedEmbedder,
)
from copilot.embeddings.fastembed_embedder import _default_factory

MODEL = "jinaai/jina-embeddings-v2-base-code"
DIM = 8


def make_tokenizer(max_length: int = 5) -> Tokenizer:
    """Word-level tokenizer with truncation and padding, like fastembed's."""
    vocab = {"[UNK]": 0, "[PAD]": 1, **{w: i + 2 for i, w in enumerate("abcdefghij")}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=1, pad_token="[PAD]")
    return tok


class FakeInner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class FakeTextEmbedding:
    """Mimics the parts of fastembed.TextEmbedding that the adapter uses."""

    supported = [
        {"model": MODEL, "dim": DIM, "size_in_GB": 0.64},
        {"model": "BAAI/bge-small-en-v1.5", "dim": 384, "size_in_GB": 0.067},
    ]
    instances: list["FakeTextEmbedding"] = []
    fail_on_init: Exception | None = None
    output = staticmethod(lambda texts: [np.full(DIM, 1.0 + i) for i in range(len(texts))])

    def __init__(self, model_name, cache_dir=None, threads=None, **kwargs):
        if FakeTextEmbedding.fail_on_init:
            raise FakeTextEmbedding.fail_on_init
        self.model_name, self.cache_dir, self.threads = model_name, cache_dir, threads
        self.model = FakeInner(make_tokenizer())
        self.calls: list[tuple[str, list[str], int | None]] = []
        FakeTextEmbedding.instances.append(self)

    @classmethod
    def list_supported_models(cls):
        return cls.supported

    def embed(self, texts, batch_size=256, **kwargs):
        texts = list(texts)
        self.calls.append(("embed", texts, batch_size))
        yield from type(self).output(texts)

    def query_embed(self, query, **kwargs):
        texts = [query] if isinstance(query, str) else list(query)
        self.calls.append(("query_embed", texts, kwargs.get("batch_size")))
        yield from type(self).output(texts)


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeTextEmbedding.instances = []
    FakeTextEmbedding.fail_on_init = None
    FakeTextEmbedding.output = staticmethod(
        lambda texts: [np.full(DIM, 1.0 + i) for i in range(len(texts))]
    )
    yield


@pytest.fixture
def embedder(tmp_path):
    return FastEmbedEmbedder(
        MODEL, cache_dir=tmp_path / "cache", batch_size=4, model_factory=lambda: FakeTextEmbedding
    )


def test_loading_is_lazy(embedder):
    assert FakeTextEmbedding.instances == []
    embedder.embed_documents(["a b"])
    assert len(FakeTextEmbedding.instances) == 1
    embedder.embed_documents(["c"])
    assert len(FakeTextEmbedding.instances) == 1  # loaded once


def test_model_is_created_with_our_cache_dir_and_threads(tmp_path):
    e = FastEmbedEmbedder(
        MODEL, cache_dir=tmp_path / "cache", threads=3, model_factory=lambda: FakeTextEmbedding
    )
    _ = e.info
    inst = FakeTextEmbedding.instances[0]
    assert inst.cache_dir == str(tmp_path / "cache") and inst.threads == 3
    assert (tmp_path / "cache").is_dir()  # created for the caller


def test_info_reports_model_dimension_runtime_and_limits(embedder):
    info = embedder.info
    assert info.model_id == MODEL
    assert info.dimension == DIM  # from the fastembed registry
    assert info.runtime == "fastembed-onnx"
    assert info.normalized is True and info.pooling == "mean" and info.asymmetric is False
    assert info.batch_size == 4
    assert info.max_input_tokens == 5  # tokenizer truncation length


def test_one_vector_per_input_in_order_with_unit_length(embedder):
    out = embedder.embed_documents(["a", "b", "c"])
    assert out.shape == (3, DIM) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.isfinite(out).all()


def test_order_is_preserved_across_multiple_batches(embedder):
    FakeTextEmbedding.output = staticmethod(
        lambda texts: [np.eye(DIM)[int(t.split()[-1]) % DIM] + 0.0 for t in texts]
    )
    texts = [f"t {i}" for i in range(11)]  # 11 texts, batch size 4 -> the runtime sees one call
    out = embedder.embed_documents(texts)
    assert [int(np.argmax(row)) for row in out] == [i % DIM for i in range(11)]
    assert FakeTextEmbedding.instances[0].calls[0][2] == 4  # batch_size is passed through


def test_empty_batch_returns_empty_array_without_calling_the_model(embedder):
    out = embedder.embed_documents([])
    assert out.shape == (0, DIM)
    assert FakeTextEmbedding.instances[0].calls == []


def test_invalid_inputs_are_rejected_before_the_model_runs(embedder):
    for bad in ("bare string", ["ok", ""], ["ok", None]):
        with pytest.raises(EmbeddingInputError):
            embedder.embed_documents(bad)
    with pytest.raises(EmbeddingInputError):
        embedder.embed_query("   ")
    assert not FakeTextEmbedding.instances or FakeTextEmbedding.instances[0].calls == []


def test_query_uses_query_embed_and_matches_document_dimension(embedder):
    doc = embedder.embed_documents(["hello"])
    query = embedder.embed_query("hello")
    assert query.shape == (DIM,) and doc.shape == (1, DIM)
    kinds = [c[0] for c in FakeTextEmbedding.instances[0].calls]
    assert kinds == ["embed", "query_embed"]


def test_non_normalised_model_output_is_normalised_by_the_adapter(embedder):
    FakeTextEmbedding.output = staticmethod(lambda texts: [np.full(DIM, 7.0) for _ in texts])
    out = embedder.embed_documents(["x"])
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda texts: [np.ones(DIM + 1) for _ in texts], "shape"),
        (lambda texts: [np.ones(DIM)], "shape"),  # wrong count for 2 inputs
        (lambda texts: [np.full(DIM, np.nan) for _ in texts], "non-finite"),
        (lambda texts: [np.zeros(DIM) for _ in texts], "all-zero"),
    ],
)
def test_bad_model_output_is_a_runtime_error(embedder, output, message):
    FakeTextEmbedding.output = staticmethod(output)
    with pytest.raises(EmbeddingRuntimeError, match=message):
        embedder.embed_documents(["a", "b"])


def test_exception_inside_the_runtime_is_wrapped_and_keeps_its_cause(embedder):
    def boom(texts):
        raise RuntimeError("onnx exploded")

    FakeTextEmbedding.output = staticmethod(boom)
    with pytest.raises(EmbeddingRuntimeError, match="onnx exploded") as info:
        embedder.embed_documents(["a"])
    assert isinstance(info.value.__cause__, RuntimeError)


def test_unsupported_model_is_a_config_error_naming_alternatives(tmp_path):
    e = FastEmbedEmbedder(
        "BAAI/bge-small-en", cache_dir=tmp_path, model_factory=lambda: FakeTextEmbedding
    )
    with pytest.raises(EmbeddingConfigError, match="not supported") as info:
        _ = e.info
    assert "bge-small-en-v1.5" in str(info.value)  # a similar supported model is suggested
    assert FakeTextEmbedding.instances == []  # nothing was downloaded/loaded


def test_download_or_load_failure_is_wrapped_with_actionable_context(embedder, tmp_path):
    FakeTextEmbedding.fail_on_init = OSError("403 Forbidden")
    with pytest.raises(EmbeddingModelUnavailableError) as info:
        _ = embedder.info
    message = str(info.value)
    assert MODEL in message and "403 Forbidden" in message and "huggingface.co" in message
    assert str(tmp_path / "cache") in message
    assert isinstance(info.value.__cause__, OSError)  # original cause preserved


def test_failed_load_can_be_retried(embedder):
    FakeTextEmbedding.fail_on_init = OSError("temporary")
    with pytest.raises(EmbeddingModelUnavailableError):
        _ = embedder.info
    FakeTextEmbedding.fail_on_init = None
    assert embedder.info.dimension == DIM


def test_invalid_batch_size_is_a_config_error(tmp_path):
    with pytest.raises(EmbeddingConfigError):
        FastEmbedEmbedder(MODEL, cache_dir=tmp_path, batch_size=0)


def test_missing_fastembed_is_a_config_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` fail
    with pytest.raises(EmbeddingConfigError, match="not installed"):
        _default_factory()


# ---- token counting ---------------------------------------------------------------------------
def test_count_tokens_is_not_truncated_and_leaves_the_embedding_tokenizer_alone(embedder):
    text = "a b c d e f g h i j"  # 10 word tokens; the embedding tokenizer truncates at 5
    assert embedder.count_tokens([text, "a"]) == [10, 1]
    original = FakeTextEmbedding.instances[0].model.tokenizer
    assert original.truncation["max_length"] == 5  # fastembed's own tokenizer still truncates
    assert original.padding is not None
    assert len(original.encode(text).ids) == 5  # ... and still produces truncated input


def test_count_tokens_rejects_invalid_input(embedder):
    with pytest.raises(EmbeddingInputError):
        embedder.count_tokens("a bare string")


def test_count_tokens_without_exposed_tokenizer_is_a_clear_error(embedder):
    _ = embedder.info
    FakeTextEmbedding.instances[0].model = None
    with pytest.raises(EmbeddingRuntimeError, match="does not expose the tokenizer"):
        embedder.count_tokens(["a"])


def test_runtime_limits_without_model_dir_are_reported_as_none(embedder):
    limits = embedder.runtime_limits()
    assert limits == {
        "tokenizer_truncation_max_length": 5,
        "config_max_position_embeddings": None,
        "tokenizer_config_model_max_length": None,
    }


def test_factory_builds_the_embedder_described_by_settings(tmp_path):
    from copilot.config.settings import Settings
    from copilot.embeddings import create_embedder

    settings = Settings(
        _env_file=None,
        embedding_model=MODEL,
        embedding_batch_size=7,
        embedding_threads=2,
        embedding_cache_dir=tmp_path / "models",
    )
    embedder = create_embedder(settings)
    assert isinstance(embedder, FastEmbedEmbedder)
    assert FakeTextEmbedding.instances == []  # constructing does not load the model
    assert embedder._batch_size == 7 and embedder._cache_dir == tmp_path / "models"
