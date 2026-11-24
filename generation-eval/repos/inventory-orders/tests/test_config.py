import pytest

from invorders.config import load_settings
from invorders.errors import ConfigError


def test_defaults():
    settings = load_settings({})
    assert settings.low_stock_threshold == 5
    assert settings.tax_rate_bps == 800


def test_overrides_and_empty_values():
    settings = load_settings({"INVORDERS_TAX_RATE_BPS": "1000", "INVORDERS_CURRENCY": ""})
    assert settings.tax_rate_bps == 1000
    assert settings.currency == "USD"


def test_unknown_variable_is_rejected():
    with pytest.raises(ConfigError):
        load_settings({"INVORDERS_TYPO": "1"})


def test_out_of_range_is_rejected():
    with pytest.raises(ConfigError):
        load_settings({"INVORDERS_TAX_RATE_BPS": "9999"})


def test_other_prefixes_are_ignored():
    assert load_settings({"PATH": "/bin"}).currency == "USD"
