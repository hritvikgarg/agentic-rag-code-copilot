import pytest

from invorders.errors import ValidationError
from invorders.validation import (
    validate_email,
    validate_postcode,
    validate_price_cents,
    validate_quantity,
    validate_sku,
)


def test_sku_accepts_canonical_form():
    assert validate_sku("ABC-0001") == "ABC-0001"


@pytest.mark.parametrize("bad", ["abc-0001", "AB-0001", "ABC0001", "ABC-001", "ABC-00001", ""])
def test_sku_rejects_malformed(bad):
    with pytest.raises(ValidationError):
        validate_sku(bad)


def test_quantity_bounds_and_types():
    assert validate_quantity(1, maximum=5) == 1
    for bad in (0, -1, 6):
        with pytest.raises(ValidationError):
            validate_quantity(bad, maximum=5)
    with pytest.raises(ValidationError):
        validate_quantity(True, maximum=5)
    with pytest.raises(ValidationError):
        validate_quantity(2.0, maximum=5)  # type: ignore[arg-type]


def test_price_must_be_integer_cents():
    assert validate_price_cents(0) == 0
    with pytest.raises(ValidationError):
        validate_price_cents(-1)
    with pytest.raises(ValidationError):
        validate_price_cents(12.5)  # type: ignore[arg-type]


def test_postcode_strips_whitespace_but_needs_five_digits():
    assert validate_postcode(" 12345 ") == "12345"
    with pytest.raises(ValidationError):
        validate_postcode("1234")


def test_email_is_lowercased():
    assert validate_email("  Ann@Example.COM ") == "ann@example.com"
    with pytest.raises(ValidationError):
        validate_email("not-an-email")
