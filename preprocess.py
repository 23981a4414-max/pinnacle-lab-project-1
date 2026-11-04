"""Text cleaning utilities for the Fake News Detection project."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<.*?>")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case, strip URLs/HTML/punctuation/digits and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
