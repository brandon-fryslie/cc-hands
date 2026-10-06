"""Where a phone reaches hands' page, and what it is reached with: this machine's tailnet name and LAN addresses, the
certificate shown under each, and the phone's key.

A browser lets a page use the microphone only over HTTPS, so hands serves nothing else. Under its tailnet name it shows
the certificate Tailscale issues for that name, which the phone trusts as it is; under any other address, a LAN one,
it shows a certificate of its own, which the phone is asked once to accept.

The phone's key is a secret kept in the home and carried in the fragment of the page's address, which a browser never
sends with the page's request.
"""

import datetime
import hmac
import ipaddress
import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

import ifaddr
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from hands.sessions.child import run
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

# All of this machine's IPv4 addresses, the LAN's and the tailnet's alike.
PHONE_HOST = "0.0.0.0"
PHONE_PORT = 47616
# How long `tailscale` may take to say this machine's name and renew its certificate, which it does online when due.
TAILSCALE_TIMEOUT_SECONDS = 30.0
# hands' own certificate is made again at a start this close to its end, rather than shown once it has ended.
RENEW_DAYS = 30


@dataclass(frozen=True)
class Tailnet:
    """This machine's name on the tailnet, and the certificate Tailscale issued for it."""

    name: str
    cert: Path
    key: Path


@dataclass(frozen=True)
class Untailed:
    """Why hands has no tailnet name to serve the page under: said in the log, and the page served on the LAN alone."""

    reason: str


async def tailnet_name() -> str | Untailed:
    """This machine's name on the tailnet, as Tailscale's own command says it."""
    status = await _tailscale("status", "--json")
    match status:
        case Untailed():
            return status
        case str():
            pass
    try:
        described: object = json.loads(status)
    except json.JSONDecodeError as error:
        return Untailed(f"tailscale status said something that is not JSON: {error}")
    match described:
        case {"CertDomains": [str(name), *_]}:  # pyright: ignore[reportUnknownVariableType]  (the first name is all that is read)
            return name
        case _:
            return Untailed("the tailnet has HTTPS certificates turned off, so Tailscale issues none for this machine")


async def tailnet(home: Home) -> Tailnet | Untailed:
    """This machine's tailnet name and a current certificate for it, from Tailscale's own command."""
    name = await tailnet_name()
    match name:
        case Untailed():
            return name
        case str():
            pass
    found = Tailnet(name, home.phone / "tailnet.crt", home.phone / "tailnet.key")
    home.phone.mkdir(parents=True, exist_ok=True)
    # Tailscale keeps the certificate and renews it when it is due, so asking again is cheap.
    issued = await _tailscale("cert", "--cert-file", str(found.cert), "--key-file", str(found.key), name)
    match issued:
        case Untailed():
            return issued
        case str():
            return found


async def _tailscale(subcommand: str, *arguments: str) -> str | Untailed:
    """What `tailscale subcommand` said, or why it said nothing."""
    # [LAW:one-source-of-truth] the one place the command is found: every asking runs the same tailscale.
    command = shutil.which("tailscale")
    if command is None:
        return Untailed("the tailscale command is not on the PATH")
    asked = f"tailscale {subcommand}"
    try:
        # [LAW:single-enforcer] ended where every child of hands is: killed and reaped on a timeout, and on a shutdown.
        ran = await run(command, subcommand, *arguments, timeout=TAILSCALE_TIMEOUT_SECONDS)
    except TimeoutError:
        return Untailed(f"{asked} took over {TAILSCALE_TIMEOUT_SECONDS:.0f}s")
    except OSError as error:
        return Untailed(f"cannot run {asked}: {error}")
    if ran.returncode != 0:
        return Untailed(f"{asked} failed ({ran.returncode}): {ran.err.decode().strip()}")
    return ran.out.decode()


def _private(path: Path, data: bytes) -> Path:
    """`data` written whole to a new file beside `path`, readable by the user alone, for a rename or a link to put in
    place: a file of the phone's is never seen half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    with os.fdopen(os.open(written, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as file:
        file.write(data)
    return written


def own_certificate(home: Home, now: datetime.datetime) -> tuple[Path, Path]:
    """The certificate hands shows under any address but its tailnet name: made once, kept in the home, and made
    again at a start that finds it within RENEW_DAYS of its end."""
    cert, key = home.phone / "own.crt", home.phone / "own.key"
    if cert.exists() and key.exists():
        ends = x509.load_pem_x509_certificate(cert.read_bytes()).not_valid_after_utc
        if ends - now > datetime.timedelta(days=RENEW_DAYS):
            return cert, key
    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "hands")])
    made = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        # As long as a browser lets a certificate it was asked to accept live; a new one is a new prompt on the phone.
        .not_valid_after(now + datetime.timedelta(days=397))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("hands"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]), critical=False)
        .sign(private, hashes.SHA256())
    )
    # The key goes in first: a start stopped between the two leaves the old certificate, near its end, which the next
    # start makes again, key and all.
    os.replace(_private(key, private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())), key)
    os.replace(_private(cert, made.public_bytes(serialization.Encoding.PEM)), cert)
    return cert, key


def phone_key(home: Home) -> str:
    """The phone's key: made once, by whichever of hands and `hands phone` asks first, kept in the home, readable by the
    user alone; raises Rejected where the file holds none."""
    path = home.phone / "key"
    if not path.exists():
        made = _private(path, secrets.token_urlsafe(24).encode())
        try:
            # [LAW:one-source-of-truth] a link, unlike a rename, never replaces: of two keys made at once, the first stands.
            os.link(made, path)
        except FileExistsError:
            pass
        finally:
            made.unlink()
    # [LAW:parse-dont-validate] a blank key would let in an offer that carries none.
    match path.read_text().strip():
        case "":
            raise Rejected(f"{path} holds no key; delete it, and hands makes a new one")
        case key:
            return key


def carries_key(authorization: str, key: str) -> bool:
    """Whether a request's Authorization header carries the phone's key as a bearer token.

    [LAW:single-enforcer] the one check of the key, for every route that asks for it, compared in constant time.
    """
    return hmac.compare_digest(authorization.removeprefix("Bearer ").encode(), key.encode())


def lan_addresses() -> list[str]:
    """This machine's IPv4 addresses on its Ethernet and Wi-Fi, which macOS names en0, en1, and on: a VPN's tunnel, the
    tailnet's among them, is no address a phone on the LAN reaches; the page is reached on the tailnet by name."""
    return [
        str(ip.ip)
        for adapter in ifaddr.get_adapters()
        if adapter.name.startswith("en")
        for ip in adapter.ips
        if ip.is_IPv4 and not ipaddress.IPv4Address(str(ip.ip)).is_link_local
    ]


def page_urls(name: str | Untailed, lan: list[str], key: str) -> list[str]:
    """Every address the phone can open the page at, the tailnet's name first, each carrying the key in its fragment."""
    match name:
        case str():
            hosts = [name, *lan]
        case Untailed():
            hosts = lan
    return [f"https://{host}:{PHONE_PORT}/#{key}" for host in hosts]
