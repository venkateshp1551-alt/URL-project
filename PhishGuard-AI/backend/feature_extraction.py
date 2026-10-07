"""Extract consistent numerical features from URL text without network access."""

import ipaddress
import re
from urllib.parse import SplitResult, urlsplit

FEATURE_NAMES = (
    "url_length",
    "domain_length",
    "number_of_dots",
    "number_of_hyphens",
    "number_of_underscores",
    "number_of_slashes",
    "number_of_question_marks",
    "number_of_equals",
    "number_of_at_symbols",
    "number_of_digits",
    "number_of_special_characters",
    "number_of_subdomains",
    "uses_https",
    "has_ip_address",
    "uses_url_shortener",
    "contains_login",
    "contains_verify",
    "contains_account",
    "contains_secure",
    "contains_update",
    "contains_password",
    "digit_ratio",
    "special_character_ratio",
)

URL_SHORTENERS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
    }
)
_EXPLICIT_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _parse_url_text(url: str) -> SplitResult | None:
    """Parse an absolute URL or a URL without its http(s) scheme."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme:
            if parsed.scheme.lower() in {"http", "https"}:
                return parsed
            return None

        if _EXPLICIT_SCHEME.match(url):
            return None

        candidate = url if url.startswith("//") else f"//{url}"
        return urlsplit(candidate)
    except ValueError:
        # Malformed authorities, such as invalid bracketed IPv6, have no host.
        return None


def _is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _is_url_shortener(hostname: str) -> bool:
    normalized_hostname = hostname.rstrip(".").lower()
    return any(
        normalized_hostname == domain
        or normalized_hostname.endswith(f".{domain}")
        for domain in URL_SHORTENERS
    )


def extract_features(url: str) -> dict[str, int | float]:
    """Return numerical URL-text features in the stable ``FEATURE_NAMES`` order.

    This function only parses and counts characters in ``url``. It never makes
    a network request or executes URL content.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    url_text = url.strip()
    parsed = _parse_url_text(url_text)
    hostname = parsed.hostname.lower().rstrip(".") if parsed and parsed.hostname else ""
    is_ip_address = _is_ip_address(hostname)
    host_labels = hostname.split(".") if hostname else []
    number_of_subdomains = (
        max(len(host_labels) - 2, 0) if host_labels and not is_ip_address else 0
    )

    url_length = len(url_text)
    number_of_digits = sum(character.isdigit() for character in url_text)
    number_of_special_characters = sum(
        not character.isalnum() for character in url_text
    )
    lower_url = url_text.lower()

    values: dict[str, int | float] = {
        "url_length": url_length,
        "domain_length": len(hostname),
        "number_of_dots": url_text.count("."),
        "number_of_hyphens": url_text.count("-"),
        "number_of_underscores": url_text.count("_"),
        "number_of_slashes": url_text.count("/"),
        "number_of_question_marks": url_text.count("?"),
        "number_of_equals": url_text.count("="),
        "number_of_at_symbols": url_text.count("@"),
        "number_of_digits": number_of_digits,
        "number_of_special_characters": number_of_special_characters,
        "number_of_subdomains": number_of_subdomains,
        "uses_https": int(bool(parsed and parsed.scheme.lower() == "https")),
        "has_ip_address": int(is_ip_address),
        "uses_url_shortener": int(_is_url_shortener(hostname)),
        "contains_login": int("login" in lower_url),
        "contains_verify": int("verify" in lower_url),
        "contains_account": int("account" in lower_url),
        "contains_secure": int("secure" in lower_url),
        "contains_update": int("update" in lower_url),
        "contains_password": int("password" in lower_url),
        "digit_ratio": number_of_digits / url_length if url_length else 0.0,
        "special_character_ratio": (
            number_of_special_characters / url_length if url_length else 0.0
        ),
    }
    return {feature_name: values[feature_name] for feature_name in FEATURE_NAMES}


if __name__ == "__main__":
    test_urls = (
        "https://www.google.com",
        "https://github.com",
        "http://example.com/login",
        "http://192.168.1.1/login",
        "http://secure-login-example.xyz/verify/account",
    )
    for test_url in test_urls:
        print(f"\nURL: {test_url}")
        for feature_name, feature_value in extract_features(test_url).items():
            print(f"  {feature_name}: {feature_value}")
