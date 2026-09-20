"""Synthetic credential builders for the secret-scanner tests.

Every "secret" here is generated at run time from a fixed seed, so the test sources never contain a
complete token-shaped literal (the repository scans itself in ``test_self_scan``). The values are
random-looking but **not real credentials**; they are deterministic so failures are reproducible.
"""

from __future__ import annotations

import hashlib
import string

BASE62 = string.ascii_letters + string.digits
UPPER_DIGITS = string.ascii_uppercase + string.digits
URL_SAFE = string.ascii_letters + string.digits + "_-"
HEX = "0123456789abcdef"
BASE64 = string.ascii_letters + string.digits + "+/"


def synth(seed: str, length: int, alphabet: str = BASE62) -> str:
    """Deterministic pseudo-random string of ``length`` characters from ``alphabet``."""
    out: list[str] = []
    counter = 0
    while len(out) < length:
        digest = hashlib.sha256(f"synthetic-{seed}-{counter}".encode()).digest()
        out.extend(alphabet[b % len(alphabet)] for b in digest)
        counter += 1
    return "".join(out[:length])


def github_token(seed: str = "gh") -> str:
    return "ghp_" + synth(seed, 36)


def aws_access_key(seed: str = "aws") -> str:
    return "AKIA" + synth(seed, 16, UPPER_DIGITS)


def google_api_key(seed: str = "goog") -> str:
    return "AIza" + synth(seed, 35, URL_SAFE)


def slack_token(seed: str = "slack") -> str:
    return "xoxb-" + synth(seed + "a", 12, string.digits) + "-" + synth(seed, 24)


def slack_webhook(seed: str = "hook") -> str:
    return (
        "https://hooks.slack.com/services/T"
        + synth(seed + "t", 9, UPPER_DIGITS)
        + "/B"
        + synth(seed + "b", 9, UPPER_DIGITS)
        + "/"
        + synth(seed, 24)
    )


def stripe_key(seed: str = "stripe") -> str:
    return "sk_live_" + synth(seed, 28)


def gitlab_token(seed: str = "gl") -> str:
    return "glpat-" + synth(seed, 24, URL_SAFE)


def huggingface_token(seed: str = "hf") -> str:
    return "hf_" + synth(seed, 36)


def sendgrid_key(seed: str = "sg") -> str:
    return "SG." + synth(seed + "1", 22, URL_SAFE) + "." + synth(seed + "2", 43, URL_SAFE)


def ai_provider_key(seed: str = "ai") -> str:
    return "sk-" + synth(seed, 48)


def anthropic_style_key(seed: str = "ant") -> str:
    return "sk-ant-" + synth(seed, 48, URL_SAFE)


def random_value(seed: str = "rv", length: int = 40) -> str:
    return synth(seed, length)


def private_key_text(kind: str = "RSA ") -> str:
    """A PEM-shaped private key block with random (non-key) base64 lines."""
    header = "-----BEGIN " + kind + "PRIVATE KEY-----"
    footer = "-----END " + kind + "PRIVATE KEY-----"
    body = [synth(f"pem{i}", 64, BASE64) for i in range(4)]
    return "\n".join([header, *body, footer]) + "\n"


def contains_secret(haystack: str, secret: str) -> bool:
    """True if the secret, or a long slice of it, appears in ``haystack``."""
    middle = secret[3:-3] if len(secret) > 12 else secret
    return secret in haystack or middle in haystack
