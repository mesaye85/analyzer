"""Deterministic IOC extraction from plain text.

Standard library only. Everything here is pattern matching plus structural
validation: it finds strings that look like indicators of compromise, not
strings that are known to be malicious.
"""

from __future__ import annotations

import bisect
import ipaddress
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

# Output order of IOC types.
IOC_TYPES = ("ipv4", "ipv6", "url", "domain", "email", "md5", "sha1", "sha256")

HASH_TYPES = {32: "md5", 40: "sha1", 64: "sha256"}

# A dotted token whose last label is one of these is treated as a file name,
# not a domain (report.txt, malware.exe). Not applied to URL hosts.
FILE_EXTENSIONS = frozenset(
    """
    txt exe dll pdf doc docx xls xlsx zip rar 7z js py sh bat ps1 png jpg jpeg
    gif html htm php json xml csv log tmp bin iso msi jar apk vbs lnk scr sys
    """.split()
)


@dataclass(frozen=True)
class IOC:
    ioc_type: str
    value: str
    note: str = ""


# --------------------------------------------------------------------------
# Refanging
# --------------------------------------------------------------------------

_DEFANG = re.compile(
    r"\bhxxp(s?)(?=\[?:)"  # hxxp:// hxxps:// hxxp[:]// hxxp[://]
    r"|\[\.\]|\(\.\)|\{\.\}|\[dot\]"  # [.] (.) {.} [dot]
    r"|\[://\]|\[:\]"  # [://] [:]
    r"|\[@\]|\[at\]",  # [@] [at]
    re.IGNORECASE,
)


def _defang_replacement(match: "re.Match[str]") -> str:
    token = match.group(0).lower()
    if token.startswith("hxxp"):
        return "http" + match.group(1).lower()
    if token in ("[.]", "(.)", "{.}", "[dot]"):
        return "."
    if token == "[://]":
        return "://"
    if token == "[:]":
        return ":"
    return "@"


def refang_with_spans(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Undo common defanging. Returns the new text and the [start, end)
    spans, in the new text, of every replacement that was made."""
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    last = 0
    pos = 0
    for match in _DEFANG.finditer(text):
        before = text[last:match.start()]
        parts.append(before)
        pos += len(before)
        replacement = _defang_replacement(match)
        parts.append(replacement)
        spans.append((pos, pos + len(replacement)))
        pos += len(replacement)
        last = match.end()
    parts.append(text[last:])
    return "".join(parts), spans


def refang(text: str) -> str:
    """Undo common defanging (hxxp, [.], (.), {.}, [dot], [:], [@], [at])."""
    return refang_with_spans(text)[0]


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_DOMAIN_BODY = rf"(?:{_LABEL}\.)+[A-Za-z]{{2,24}}"
_DOMAIN_FULL = re.compile(_DOMAIN_BODY)

_IPV4 = re.compile(
    r"(?<![0-9A-Za-z.])"
    r"(?:\d{1,3}\.){3}\d{1,3}"
    r"(?![0-9A-Za-z]|\.[0-9A-Za-z])"
)

# Candidate IPv6 token: at least two colons, '::' allowed, optional embedded
# IPv4 tail and %zone. ipaddress.IPv6Address decides whether it is real, so
# times such as 12:30:45 are rejected there.
_IPV6 = re.compile(
    r"(?<![0-9A-Za-z:])"
    r"((?:[0-9A-Fa-f]{0,4}:){2,7}(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9A-Fa-f]{1,4})?)"
    r"(?:%[0-9A-Za-z_.-]+)?"
    r"(?![0-9A-Za-z:]|\.\d)"
)

# Stops at whitespace, quotes and brackets. A bracketed IPv6 host is allowed
# directly after the scheme.
_URL = re.compile(
    r"\b(?:https?|ftp)://(?:\[[0-9A-Fa-f:.]+\])?[^\s\"'<>()\[\]{}`\\^|]*",
    re.IGNORECASE,
)
_URL_TRAILING = ".,;:!?*"

_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    rf"([A-Za-z0-9_][A-Za-z0-9_.%+-]*)@({_DOMAIN_BODY})"
    r"(?![A-Za-z0-9-]|\.[A-Za-z0-9])"
)

_HEX = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{32,}(?![0-9A-Za-z])")

# Underscores are excluded on both sides so "_dmarc.example.com" or
# "foo_bar.evil.com" yields nothing rather than a truncated name.
_DOMAIN = re.compile(
    r"(?<![A-Za-z0-9@._-])"
    rf"{_DOMAIN_BODY}"
    r"(?![A-Za-z0-9@_-]|\.[A-Za-z0-9])"
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _SpanSet:
    """Merged [start, end) intervals with a fast overlap query."""

    def __init__(self, spans: Iterable[Tuple[int, int]]):
        merged: List[List[int]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self._starts = [s for s, _ in merged]
        self._ends = [e for _, e in merged]

    def overlaps(self, start: int, end: int) -> bool:
        i = bisect.bisect_right(self._ends, start)
        return i < len(self._starts) and self._starts[i] < end


def ip_scope(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> str:
    """Classify an address using the ipaddress module's properties.

    "private" is ipaddress.is_private, which covers every range IANA marks as
    not globally reachable (RFC 1918, but also documentation ranges such as
    192.0.2.0/24 and 2001:db8::/32). "non-global" is anything left over that
    is not globally reachable either, e.g. 100.64.0.0/10 (carrier-grade NAT).
    """
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_reserved:
        return "reserved"
    if ip.is_private:
        return "private"
    if ip.is_global:
        return "global"
    return "non-global"


def is_domain(name: str, allow_file_extension: bool = False) -> bool:
    """Structural domain check: labels of 1-63 letters, digits or hyphens (no
    leading/trailing hyphen), total length <= 253, alphabetic TLD of 2-24
    characters. Bare IPs fail because the TLD must be alphabetic. There is no
    check against the list of real TLDs."""
    if len(name) > 253 or not _DOMAIN_FULL.fullmatch(name):
        return False
    if not allow_file_extension and name.rsplit(".", 1)[-1].lower() in FILE_EXTENSIONS:
        return False
    return True


def _valid_email(local: str, domain: str) -> bool:
    return len(local) <= 64 and ".." not in local and not local.endswith(".") and len(domain) <= 253


def _url_host(url: str) -> Optional[str]:
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


@dataclass
class _Hit:
    pos: int
    key: str
    value: str
    note: str
    refanged: bool


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_iocs(text: str) -> List[IOC]:
    """Extract IOCs from text.

    Results are grouped by type (in IOC_TYPES order), de-duplicated, and kept
    in first-seen order within each type. The note column carries the IP scope,
    "from url" for a domain taken from a URL host, and "refanged" when the
    indicator appeared defanged in the source text.
    """
    clean, fang_spans = refang_with_spans(text)
    fanged = _SpanSet(fang_spans)
    hits: Dict[str, List[_Hit]] = {t: [] for t in IOC_TYPES}

    def add(ioc_type: str, start: int, end: int, key: str, value: str, note: str = "") -> None:
        hits[ioc_type].append(_Hit(start, key, value, note, fanged.overlaps(start, end)))

    # URLs, and their hosts as domains.
    url_spans = []
    for match in _URL.finditer(clean):
        url = match.group(0).rstrip(_URL_TRAILING)
        host = _url_host(url)
        if not host:
            continue
        start, end = match.start(), match.start() + len(url)
        url_spans.append((start, end))
        add("url", start, end, url, url)
        if is_domain(host, allow_file_extension=True):
            offset = url.lower().find(host, url.find("//") + 2)
            h_start = start + max(offset, 0)
            add("domain", h_start, h_start + len(host), host, host, "from url")

    # Emails, except user@host credentials inside a URL.
    in_url = _SpanSet(url_spans)
    email_spans = []
    for match in _EMAIL.finditer(clean):
        local, domain = match.group(1), match.group(2)
        if not _valid_email(local, domain) or in_url.overlaps(*match.span()):
            continue
        email_spans.append(match.span())
        add("email", match.start(), match.end(), match.group(0).lower(), match.group(0))

    # IPv6 first, so an embedded IPv4 tail (::ffff:192.0.2.1) is not also
    # reported as a separate IPv4 address.
    ipv6_spans = []
    for match in _IPV6.finditer(clean):
        token = match.group(1)
        if not any(c in "0123456789abcdefABCDEF" for c in token):
            continue  # a bare "::" in prose or code
        if token.endswith(":") and not token.endswith("::") and match.group(0) == token:
            token = token[:-1]  # "2001:db8::1: the host" -> drop the sentence colon
        try:
            ip6 = ipaddress.IPv6Address(token)
        except ValueError:
            continue
        start = match.start(1)
        ipv6_spans.append((start, start + len(token)))
        add("ipv6", start, start + len(token), ip6.compressed, token.lower(), f"scope={ip_scope(ip6)}")
    in_ipv6 = _SpanSet(ipv6_spans)

    for match in _IPV4.finditer(clean):
        if in_ipv6.overlaps(*match.span()):
            continue
        try:
            ip4 = ipaddress.IPv4Address(match.group(0))
        except ValueError:
            continue
        add("ipv4", match.start(), match.end(), str(ip4), str(ip4), f"scope={ip_scope(ip4)}")

    # Hashes: exact-length hex tokens only.
    for match in _HEX.finditer(clean):
        hash_type = HASH_TYPES.get(len(match.group(0)))
        if hash_type:
            value = match.group(0).lower()
            add(hash_type, match.start(), match.end(), value, value)

    # Domains in free text. Anything inside a URL or email was handled above.
    taken = _SpanSet(url_spans + email_spans)
    for match in _DOMAIN.finditer(clean):
        if taken.overlaps(*match.span()):
            continue
        name = match.group(0).lower()
        if is_domain(name):
            add("domain", match.start(), match.end(), name, name)

    return _dedupe(hits)


def _dedupe(hits: Dict[str, List[_Hit]]) -> List[IOC]:
    results: List[IOC] = []
    for ioc_type in IOC_TYPES:
        seen: Dict[str, _Hit] = {}
        for hit in sorted(hits[ioc_type], key=lambda h: h.pos):
            first = seen.get(hit.key)
            if first is None:
                seen[hit.key] = hit
            elif hit.refanged:
                first.refanged = True
        for hit in seen.values():
            notes = [n for n in (hit.note, "refanged" if hit.refanged else "") if n]
            results.append(IOC(ioc_type, hit.value, "; ".join(notes)))
    return results
