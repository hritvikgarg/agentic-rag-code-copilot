import numpy as np
import pytest

from copilot.embeddings.errors import EmbeddingInputError, EmbeddingRuntimeError
from copilot.embeddings.vectors import finalize_vectors, validate_text, validate_texts


class TestValidateTexts:
    def test_list_is_returned_as_list(self):
        assert validate_texts(["a", "b"]) == ["a", "b"]

    def test_empty_batch_is_valid(self):
        assert validate_texts([]) == []

    def test_generators_and_tuples_are_accepted(self):
        assert validate_texts(("a", "b")) == ["a", "b"]
        assert validate_texts(t for t in ["x"]) == ["x"]

    def test_bare_string_is_rejected_not_iterated_by_character(self):
        with pytest.raises(EmbeddingInputError, match="single str"):
            validate_texts("hello")

    @pytest.mark.parametrize("bad", [None, 5, b"bytes", ["nested"]])
    def test_non_string_items_are_rejected_with_their_index(self, bad):
        with pytest.raises(EmbeddingInputError, match=r"texts\[1\]"):
            validate_texts(["ok", bad])

    @pytest.mark.parametrize("blank", ["", " ", "\n\t "])
    def test_blank_items_are_rejected(self, blank):
        with pytest.raises(EmbeddingInputError, match="empty or whitespace"):
            validate_texts(["ok", blank])

    def test_non_iterable_is_rejected(self):
        with pytest.raises(EmbeddingInputError):
            validate_texts(42)  # type: ignore[arg-type]

    def test_input_error_is_also_a_value_error(self):
        with pytest.raises(ValueError):
            validate_text("")


class TestFinalizeVectors:
    def test_normalises_to_unit_length_float32(self):
        out = finalize_vectors([[3.0, 4.0], [0.0, 2.0]], count=2, dimension=2)
        assert out.dtype == np.float32
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
        assert np.allclose(out[0], [0.6, 0.8])

    def test_is_idempotent_for_unit_vectors(self):
        unit = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert np.array_equal(finalize_vectors(unit, count=2, dimension=2), unit)

    def test_output_is_read_only(self):
        out = finalize_vectors([[1.0, 0.0]], count=1, dimension=2)
        with pytest.raises(ValueError):
            out[0, 0] = 5.0

    def test_empty_batch_has_defined_shape(self):
        out = finalize_vectors([], count=0, dimension=768)
        assert out.shape == (0, 768) and out.dtype == np.float32

    @pytest.mark.parametrize("raw", [[[1.0, 2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0]])
    def test_wrong_shape_is_a_runtime_error(self, raw):
        with pytest.raises(EmbeddingRuntimeError, match="shape"):
            finalize_vectors(raw, count=1, dimension=2)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_are_a_runtime_error(self, bad):
        with pytest.raises(EmbeddingRuntimeError, match="non-finite"):
            finalize_vectors([[1.0, bad]], count=1, dimension=2)

    def test_zero_vector_is_a_runtime_error(self):
        with pytest.raises(EmbeddingRuntimeError, match="all-zero"):
            finalize_vectors([[0.0, 0.0]], count=1, dimension=2)

    def test_unconvertible_output_is_a_runtime_error(self):
        with pytest.raises(EmbeddingRuntimeError, match="converted"):
            finalize_vectors([["a", "b"]], count=1, dimension=2)
