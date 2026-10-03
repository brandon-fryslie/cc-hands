"""The phone's page: served over HTTPS to the LAN and the tailnet, where a phone's browser opens it and calls hands.

A browser lets a page use the microphone only over HTTPS, so hands serves nothing else. Under its tailnet name it shows
the certificate Tailscale issues for that name, which the phone trusts as it is; under any other address, a LAN one,
it shows a certificate of its own, which the phone is asked once to accept.

The page is open to anyone who can reach the port; a call is not. The phone's key is a secret kept in the home and
carried in the fragment of the page's address, which a browser never sends with the page's request: the page sends it
with its offer, and an offer without it is refused.
"""

import asyncio
import datetime
import hmac
import ipaddress
import json
import secrets
import shutil
import ssl
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import ifaddr
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from loguru import logger

from hands.sessions.audit import PhoneRefused, PhoneServing, PhoneUntailed, Record
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.phone import Offer, Phone

# All of this machine's IPv4 addresses, the LAN's and the tailnet's alike.
PHONE_HOST = "0.0.0.0"
PHONE_PORT = 47616
# How long `tailscale` may take to say this machine's name and renew its certificate, which it does online when due.
TAILSCALE_TIMEOUT_SECONDS = 30.0


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


async def tailnet(home: Home) -> Tailnet | Untailed:
    """This machine's tailnet name and a current certificate for it, from Tailscale's own command."""
    command = shutil.which("tailscale")
    if command is None:
        return Untailed("the tailscale command is not on the PATH")
    status = await _run(command, "status", "--json")
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
            pass
        case _:
            return Untailed("the tailnet has HTTPS certificates turned off, so Tailscale issues none for this machine")
    folder = home.phone
    folder.mkdir(parents=True, exist_ok=True)
    found = Tailnet(name, folder / "tailnet.crt", folder / "tailnet.key")
    # Tailscale keeps the certificate and renews it when it is due, so asking again at every start is cheap.
    issued = await _run(command, "cert", "--cert-file", str(found.cert), "--key-file", str(found.key), name)
    match issued:
        case Untailed():
            return issued
        case str():
            return found


async def _run(*command: str) -> str | Untailed:
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(), TAILSCALE_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        return Untailed(f"{' '.join(command[1:3])} took over {TAILSCALE_TIMEOUT_SECONDS:.0f}s")
    if process.returncode != 0:
        return Untailed(f"{' '.join(command[1:3])} failed ({process.returncode}): {err.decode().strip()}")
    return out.decode()


def own_certificate(home: Home) -> tuple[Path, Path]:
    """The certificate hands shows under any address but its tailnet name: made once, kept in the home."""
    cert, key = home.phone / "own.crt", home.phone / "own.key"
    if cert.exists() and key.exists():
        return cert, key
    home.phone.mkdir(parents=True, exist_ok=True)
    private = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "hands")])
    now = datetime.datetime.now(datetime.UTC)
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
    key.touch(mode=0o600)
    key.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert.write_bytes(made.public_bytes(serialization.Encoding.PEM))
    return cert, key


def tls(home: Home, net: Tailnet | Untailed) -> ssl.SSLContext:
    """The server's TLS: the tailnet's certificate for its name, hands' own for every other."""
    own = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    own.load_cert_chain(*own_certificate(home))
    match net:
        case Untailed():
            pass
        case Tailnet(name=name, cert=cert, key=key):
            tailed = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tailed.load_cert_chain(cert, key)

            def chosen(connection: ssl.SSLObject, server_name: str | None, _context: ssl.SSLContext) -> None:
                # [LAW:single-enforcer] the one place a name picks its certificate: the name the phone asked for.
                if server_name == name:
                    connection.context = tailed

            own.sni_callback = chosen  # pyright: ignore[reportAttributeAccessIssue]  (typeshed types the callback for SSLSocket alone)
    return own


def phone_key(home: Home) -> str:
    """The phone's key: made once, kept in the home, readable by the user alone."""
    path = home.phone / "key"
    if not path.exists():
        home.phone.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(24))
    return path.read_text().strip()


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


def page_urls(net: Tailnet | Untailed, lan: list[str], key: str) -> list[str]:
    """Every address the phone can open the page at, the tailnet's first, each carrying the key in its fragment."""
    match net:
        case Tailnet(name=name):
            hosts = [name, *lan]
        case Untailed():
            hosts = lan
    return [f"https://{host}:{PHONE_PORT}/#{key}" for host in hosts]


def parse_offer(body: object) -> Offer:
    """A page's offer as hands takes it; raises Rejected naming what is wrong with it."""
    match body:
        case {"sdp": str(sdp), "type": "offer", "rate": int(rate)} if rate > 0:
            return Offer(sdp, "offer", rate)
        case _:
            raise Rejected("an offer is {sdp, type: offer, rate}")


def phone_app(phone: Phone, key: str, record: Record) -> web.Application:
    """The page, and the one route that takes its calls."""
    page = resources.files("hands.voice").joinpath("phone.html").read_text()

    async def shown(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def offered(request: web.Request) -> web.Response:
        remote = request.remote or "unknown"
        given = request.headers.get("Authorization", "").removeprefix("Bearer ")
        # [LAW:single-enforcer] the one check of the key, compared in constant time.
        if not hmac.compare_digest(given.encode(), key.encode()):
            record(PhoneRefused(remote=remote))
            return web.Response(status=401, text="this page's address is missing the phone's key; open it from `hands phone`")
        try:
            offer = parse_offer(await request.json())
        except (Rejected, json.JSONDecodeError) as error:
            return web.Response(status=400, text=str(error))
        answer = await phone.answer(offer, remote)
        return web.json_response({"sdp": answer.sdp, "type": answer.type})

    app = web.Application()
    app.router.add_get("/", shown)
    app.router.add_post("/offer", offered)
    return app


async def serve_phone(phone: Phone, home: Home, record: Record) -> web.AppRunner:
    """Serve the page and take its calls on every address this machine has, until the returned runner is cleaned up."""
    net = await tailnet(home)
    app = phone_app(phone, phone_key(home), record)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, PHONE_HOST, PHONE_PORT, ssl_context=tls(home, net)).start()
    except OSError as error:
        await runner.cleanup()
        raise RuntimeError(f"cannot serve the phone's page on port {PHONE_PORT}: {error}") from error
    match net:
        case Tailnet(name=name):
            record(PhoneServing(port=PHONE_PORT, tailnet=name))
        case Untailed(reason=reason):
            # [LAW:no-silent-failure] the page is still served on the LAN; why the tailnet name is not, is said.
            logger.warning(f"the phone's page is served on the LAN alone: {reason}")
            record(PhoneUntailed(port=PHONE_PORT, reason=reason))
    return runner
