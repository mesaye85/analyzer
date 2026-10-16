import pytest

from ioc_analyzer.extract import extract_iocs, is_domain, refang


def values(text, ioc_type):
    return [i.value for i in extract_iocs(text) if i.ioc_type == ioc_type]


def notes(text, ioc_type):
    return {i.value: i.note for i in extract_iocs(text) if i.ioc_type == ioc_type}


# --- IPv4 ------------------------------------------------------------------

def test_ipv4_found():
    assert values("C2 at 8.8.8.8 and 10.0.0.5.", "ipv4") == ["8.8.8.8", "10.0.0.5"]


@pytest.mark.parametrize("text", ["999.1.1.1", "1.2.3", "1.2.3.4.5", "v1.2.3.4a", "010.1.1.1"])
def test_ipv4_rejected(text):
    assert values(text, "ipv4") == []


def test_ip_scope_tagging():
    found = notes("8.8.8.8 10.1.2.3 192.168.1.1 127.0.0.1 169.254.1.1 240.0.0.1", "ipv4")
    assert found == {
        "8.8.8.8": "scope=global",
        "10.1.2.3": "scope=private",
        "192.168.1.1": "scope=private",
        "127.0.0.1": "scope=loopback",
        "169.254.1.1": "scope=link-local",
        "240.0.0.1": "scope=reserved",
    }


def test_private_ips_are_tagged_not_filtered():
    assert "10.0.0.5" in values("internal host 10.0.0.5", "ipv4")


# --- IPv6 ------------------------------------------------------------------

def test_ipv6_compressed():
    assert values("beacon to 2001:db8::1 every 60s", "ipv6") == ["2001:db8::1"]


def test_ipv6_full():
    full = "2001:0db8:0000:0000:0000:ff00:0042:8329"
    assert values(f"addr {full}.", "ipv6") == [full]


def test_ipv6_zone_stripped_and_scoped():
    assert notes("link fe80::1%eth0 up", "ipv6") == {"fe80::1": "scope=link-local"}


def test_ipv6_global_and_loopback():
    found = notes("2606:4700:4700::1111 and ::1", "ipv6")
    assert found == {"2606:4700:4700::1111": "scope=global", "::1": "scope=loopback"}


def test_ipv6_trailing_sentence_colon():
    assert values("Host 2001:db8::1: seen twice", "ipv6") == ["2001:db8::1"]


def test_ipv4_mapped_ipv6_not_double_reported():
    text = "mapped ::ffff:192.0.2.1 here"
    assert values(text, "ipv6") == ["::ffff:192.0.2.1"]
    assert values(text, "ipv4") == []


@pytest.mark.parametrize("text", ["at 12:30:45 UTC", "10:15", "2024-01-02T12:30:45Z", "a :: b",
                                  "std::vector", "1:2:3:4:5:6:7:8:9"])
def test_ipv6_rejects_times_and_noise(text):
    assert values(text, "ipv6") == []


def test_mac_address_not_ipv6():
    assert values("MAC 00:1a:2b:3c:4d:5e", "ipv6") == []


# --- URLs ------------------------------------------------------------------

def test_url_basic():
    assert values("see https://example.com/path?a=1&b=2#frag now", "url") == [
        "https://example.com/path?a=1&b=2#frag"
    ]


@pytest.mark.parametrize("text, expected", [
    ("Visit http://example.com/a.", "http://example.com/a"),
    ("Visit http://example.com/a, then", "http://example.com/a"),
    ("(see http://example.com/a)", "http://example.com/a"),
    ('href="http://example.com/a"', "http://example.com/a"),
    ("<http://example.com/a>", "http://example.com/a"),
    ("[http://example.com/a]", "http://example.com/a"),
    ("ftp://files.example.net/drop.zip;", "ftp://files.example.net/drop.zip"),
    ("http://203.0.113.5:8080/gate.php!", "http://203.0.113.5:8080/gate.php"),
])
def test_url_boundaries(text, expected):
    assert values(text, "url") == [expected]


def test_url_rejects_other_schemes_and_bare_scheme():
    assert values("file:///etc/passwd mailto:a@b.com https:// alone", "url") == []


def test_url_host_also_reported_as_domain():
    found = notes("https://cdn.evil-site.com/x.js", "domain")
    assert found == {"cdn.evil-site.com": "from url"}


def test_url_with_ip_host_reports_ip_not_domain():
    text = "http://203.0.113.5/a"
    assert values(text, "domain") == []
    assert values(text, "ipv4") == ["203.0.113.5"]


# --- Defanging -------------------------------------------------------------

def test_refang_function():
    assert refang("hxxps://evil[.]example(.)com{.}net a[@]b[.]io c[at]d[dot]org 1[.]2[.]3[.]4 hxxp[:]//x") == (
        "https://evil.example.com.net a@b.io c@d.org 1.2.3.4 http://x"
    )


def test_defanged_url_is_refanged_and_noted():
    text = "Payload pulled from hxxps://evil[.]example[.]com/a today."
    iocs = extract_iocs(text)
    urls = [i for i in iocs if i.ioc_type == "url"]
    domains = [i for i in iocs if i.ioc_type == "domain"]
    assert [(u.value, u.note) for u in urls] == [("https://evil.example.com/a", "refanged")]
    assert [(d.value, d.note) for d in domains] == [("evil.example.com", "from url; refanged")]


def test_defanged_ip_and_email():
    assert notes("C2 198.51.100[.]23", "ipv4") == {"198.51.100.23": "scope=private; refanged"}
    assert notes("from ops[at]corp[.]net", "email") == {"ops@corp.net": "refanged"}
    assert notes("from ops[@]corp.net", "email") == {"ops@corp.net": "refanged"}


def test_plain_ioc_not_marked_refanged_by_nearby_defang():
    assert notes("evil[.]com and good.example.org", "domain") == {
        "evil.com": "refanged",
        "good.example.org": "",
    }


# --- Emails ----------------------------------------------------------------

def test_email_inside_text():
    text = "Phishing sent from billing.dept+x@pay-secure.co.uk to staff."
    assert values(text, "email") == ["billing.dept+x@pay-secure.co.uk"]


def test_email_with_underscore_kept_whole():
    assert values("reply to first_last@mail.example.com", "email") == ["first_last@mail.example.com"]


@pytest.mark.parametrize("text", ["user@localhost", "a@b", "user@domain.c", "user@exa|mple.com", "a..b@example.com"])
def test_email_rejected(text):
    assert values(text, "email") == []


def test_url_credentials_not_reported_as_email():
    text = "http://user:pass@evil.com/x"
    assert values(text, "email") == []
    assert values(text, "domain") == ["evil.com"]


def test_email_domain_not_reported_separately():
    assert values("contact bad.actor@evil-mail.com", "domain") == []


# --- Hashes ----------------------------------------------------------------

MD5 = "d41d8cd98f00b204e9800998ecf8427e"
SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_types_by_length():
    text = f"md5 {MD5} sha1 {SHA1.upper()} sha256 {SHA256}"
    assert values(text, "md5") == [MD5]
    assert values(text, "sha1") == [SHA1]
    assert values(text, "sha256") == [SHA256]


@pytest.mark.parametrize("length", [31, 33, 39, 41, 50, 63, 65, 128])
def test_hash_wrong_length_rejected(length):
    token = ("0123456789abcdef" * 10)[:length]
    iocs = extract_iocs(f"value {token} end")
    assert [i for i in iocs if i.ioc_type in ("md5", "sha1", "sha256")] == []


def test_hash_embedded_in_word_rejected():
    assert values(f"x{MD5}", "md5") == []
    assert values(f"{MD5}g", "md5") == []


# --- Domains ---------------------------------------------------------------

def test_domain_in_text():
    assert values("Resolved update-check.example.net and Example.COM.", "domain") == [
        "update-check.example.net",
        "example.com",
    ]


@pytest.mark.parametrize("name", ["report.txt", "malware.exe", "loader.dll", "invoice.pdf", "run.ps1",
                                  "index.html", "archive.7z", "a.b.json", "Dropper.EXE"])
def test_filenames_not_domains(name):
    assert values(f"dropped {name} on disk", "domain") == []


@pytest.mark.parametrize("text", ["1.2.3.4", "-bad.example.com", "bad-.example.com",
                                  "example.c0m", "example." + "a" * 25, "localhost"])
def test_domain_rejected(text):
    assert values(text, "domain") == []


def test_domain_label_and_total_length_limits():
    assert is_domain("a" * 63 + ".com")
    assert not is_domain("a" * 64 + ".com")
    assert is_domain(".".join(["a" * 60] * 4) + ".com")  # 247 chars
    assert not is_domain(".".join(["a" * 50] * 5) + ".com")  # 258 chars


@pytest.mark.parametrize("text", ["_dmarc.example.com", "foo_bar.evil.com", "evil.com_old"])
def test_domain_never_truncated_at_underscore(text):
    assert values(text, "domain") == []


def test_url_host_with_file_extension_tld_kept():
    # .zip is a real TLD; inside a URL it is clearly a host, so it is kept.
    assert values("http://update.zip/x", "domain") == ["update.zip"]


# --- Dedupe and order ------------------------------------------------------

def test_dedupe_keeps_first_seen_order():
    text = "8.8.4.4 then 1.1.1.1 then 8.8.4.4 again, EVIL.com and evil.com"
    assert values(text, "ipv4") == ["8.8.4.4", "1.1.1.1"]
    assert values(text, "domain") == ["evil.com"]


def test_empty_text():
    assert extract_iocs("") == []
    assert extract_iocs("Nothing to see here.") == []
