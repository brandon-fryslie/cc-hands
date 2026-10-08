"""hands.app's launcher, compiled and run: it starts hands only on a license key a stand-in for Polar says is live, or
offline within the grace period; the login shell says the environment, hands runs on it as the app's child, a quit winds
hands down, and a failure is told with this launch's own lines."""

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

ROOT = Path(__file__).parent.parent
ORGANIZATION = "fda84e25-7b55-4d67-916d-60ead04ff61f"
LIVE = "HANDS-1C285B2D-6CE6-4BC7-B8BE-ADB6A7E304DA"


@dataclass
class Polar:
    """A stand-in for Polar's license key validation: it answers `status` and `body`, or with `status` None closes the
    connection unanswered, as a network that fails does; it keeps each request's body."""

    url: str
    status: int | None = 200
    body: object = field(default_factory=lambda: {"status": "granted", "key": LIVE})
    asked: list[dict[str, object]] = field(default_factory=lambda: list[dict[str, object]]())


@pytest.fixture(scope="module")
def stand_in() -> Iterator[Polar]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            polar.asked.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if polar.status is None:
                self.close_connection = True
                return
            answer = json.dumps(polar.body).encode()
            self.send_response(polar.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    polar = Polar(url=f"http://127.0.0.1:{server.server_address[1]}/v1/customer-portal/license-keys/validate")
    Thread(target=server.serve_forever, daemon=True).start()
    yield polar
    server.shutdown()


@pytest.fixture
def polar(stand_in: Polar) -> Polar:
    """The stand-in, saying the live key is live, having been asked nothing yet."""
    stand_in.status, stand_in.body, stand_in.asked = 200, {"status": "granted", "key": LIVE}, []
    return stand_in


@pytest.fixture(scope="module")
def executable(tmp_path_factory: pytest.TempPathFactory, stand_in: Polar) -> Path:
    """The launcher in a bundle of its own, so its Info.plist keeps it out of the Dock as hands.app's does, selling
    through the stand-in."""
    contents = tmp_path_factory.mktemp("app") / "hands.app" / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "Info.plist", contents / "Info.plist")
    for key, value in [
        ("HandsLicenseValidate", stand_in.url),
        ("HandsLicenseOrganization", ORGANIZATION),
        ("HandsLicensePortal", "https://polar.sh/hands/portal"),
    ]:
        subprocess.run(["plutil", "-replace", key, "-string", value, contents / "Info.plist"], check=True)
    subprocess.run([ROOT / "scripts" / "swiftc-app.sh", "-o", contents / "MacOS" / "hands"], check=True)
    return contents / "MacOS" / "hands"


Open = Callable[[str, str], subprocess.Popen[bytes]]


def kept(tmp_path: Path, key: str, validated: datetime) -> None:
    """A license the app kept, last seen live at `validated`."""
    (tmp_path / "license.json").write_text(json.dumps({"key": key, "validated": validated.strftime("%Y-%m-%dT%H:%M:%SZ")}))


@pytest.fixture
def opened(executable: Path, tmp_path: Path, polar: Polar) -> Iterator[Open]:
    """Open the app on a login shell that runs `rc` before it says its environment, with `hands` on its PATH running
    `hands`; its log is tmp_path/hands.log, and the license it keeps is tmp_path/license.json, the live key unless the
    test kept another."""
    kept(tmp_path, LIVE, datetime.now(UTC))
    apps: list[subprocess.Popen[bytes]] = []

    def open_(rc: str, hands: str) -> subprocess.Popen[bytes]:
        bin_ = tmp_path / "bin"
        bin_.mkdir()
        (bin_ / "hands").write_text(f"#!/bin/sh\n{hands}\n")
        (bin_ / "hands").chmod(0o755)
        # Called as a login shell is, `-l -i -c COMMAND`: the command is $4.
        shell = tmp_path / "shell"
        shell.write_text(
            f"#!/bin/sh\necho \"an rc file's own output\"\nPATH={bin_}:/usr/bin:/bin; export PATH\nFROM_RC=yes; export FROM_RC\n{rc}\nexec /bin/sh -c \"$4\"\n"
        )
        shell.chmod(0o755)
        apps.append(
            subprocess.Popen(
                [executable],
                env={
                    **os.environ,
                    "HANDS_APP_SHELL": str(shell),
                    "HANDS_APP_LOG": str(tmp_path / "hands.log"),
                    "HANDS_APP_LICENSE": str(tmp_path / "license.json"),
                },
            )
        )
        return apps[-1]

    yield open_
    # Quit, so a hands a failed test left running is wound down with its app, not orphaned.
    for app in apps:
        app.send_signal(signal.SIGTERM)
        try:
            app.wait(timeout=5)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait()


def logged(tmp_path: Path, line: str, within: float = 10) -> str:
    """The log, once it holds `line`."""
    log = tmp_path / "hands.log"
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if log.exists() and line in (text := log.read_text()):
            return text
        time.sleep(0.05)
    raise AssertionError(f"the log never said {line!r}: {log.read_text() if log.exists() else '(no log)'}")


def test_hands_runs_on_the_login_shells_environment_and_its_clean_exit_ends_the_app(opened: Open, tmp_path: Path) -> None:
    app = opened("", 'echo "hands saw FROM_RC=$FROM_RC, run with $1, in $PWD, LANG=$LANG"; exit 0')
    assert app.wait(timeout=10) == 0
    text = logged(tmp_path, "hands exited 0")
    # The home Terminal starts in, and a locale as Terminal sets one.
    assert f"hands saw FROM_RC=yes, run with run, in {Path.home()}, LANG=" in text and ".UTF-8\n" in text
    assert "started hands run as pid" in text and f"PATH {tmp_path / 'bin'}:/usr/bin:/bin" in text


def test_a_quit_winds_hands_down_and_the_app_ends_once_it_has(opened: Open, tmp_path: Path) -> None:
    # sleep keeps SIGTERM's default: a hands that inherited the app's ignoring of it would never end.
    app = opened("", "echo up; exec sleep 600")
    logged(tmp_path, "up")
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=10) == 0
    assert "hands was ended by signal 15" in logged(tmp_path, "hands was ended")


def test_a_hands_that_does_not_end_on_sigterm_is_killed_when_the_app_is_quit_again(opened: Open, tmp_path: Path) -> None:
    app = opened("", "trap '' TERM; echo up; while :; do sleep 0.1; done")
    logged(tmp_path, "up")
    app.send_signal(signal.SIGTERM)
    with pytest.raises(subprocess.TimeoutExpired):
        app.wait(timeout=1)
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=5) == 0
    text = logged(tmp_path, "hands was ended by signal 9")
    assert "hands has not ended on SIGTERM, and the app was quit again: killing it" in text


def test_a_failed_hands_is_told_with_this_launchs_lines(opened: Open, tmp_path: Path) -> None:
    (tmp_path / "hands.log").write_text("an earlier launch's line\n")
    app = opened("", 'echo "refused: hands is already running"; exit 3')
    told = logged(tmp_path, "stopped: hands exited 3.").split("stopped: ", 1)[1]
    assert "refused: hands is already running" in told and "an earlier launch's line" not in told
    # A SIGTERM ends the app while its alert is up.
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=5) == 0


def test_a_shell_that_cannot_say_its_environment_is_told_with_its_words(opened: Open, tmp_path: Path) -> None:
    app = opened('echo "open terminal failed: not a terminal" >&2; exit 1', "exit 0")
    told = logged(tmp_path, "stopped: hands.app could not read the environment").split("stopped: ", 1)[1]
    app.send_signal(signal.SIGTERM)
    assert "it exited 1" in told and "open terminal failed: not a terminal" in told
    assert "started hands run" not in logged(tmp_path, "stopped: ")


def test_a_quit_while_the_shell_reads_its_rc_files_ends_the_app_and_the_shell(opened: Open, tmp_path: Path) -> None:
    # An interactive shell ignores SIGTERM while it reads its rc files.
    app = opened(f"echo $$ > {tmp_path / 'shell.pid'}; trap '' TERM; while :; do sleep 0.1; done", "exit 0")
    pid = tmp_path / "shell.pid"
    deadline = time.monotonic() + 10
    while not pid.exists() or not pid.read_text().strip():
        assert time.monotonic() < deadline, "the shell never started"
        time.sleep(0.05)
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=5) == 0
    deadline = time.monotonic() + 5
    while True:
        try:
            os.kill(int(pid.read_text()), 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "the shell outlived the app"
        time.sleep(0.05)


def test_a_log_past_its_limit_is_begun_again_with_the_last_kept_beside_it(opened: Open, tmp_path: Path) -> None:
    last = "an earlier launch's line\n" + "x" * 10_000_000
    (tmp_path / "hands.log").write_text(last)
    app = opened("", "exit 0")
    assert app.wait(timeout=10) == 0
    assert (tmp_path / "hands.log.1").read_text() == last
    text = logged(tmp_path, "hands exited 0")
    assert f"began this log again; the last one is {tmp_path / 'hands.log'}.1" in text and "x" * 100 not in text


def test_a_background_job_an_rc_file_starts_does_not_hold_up_hands(opened: Open, tmp_path: Path) -> None:
    # The job keeps the shell's stdout open past the shell's exit.
    app = opened(f"sleep 30 & echo $! > {tmp_path / 'job.pid'}", "exit 0")
    try:
        assert app.wait(timeout=10) == 0
        assert "hands exited 0" in logged(tmp_path, "hands exited 0")
    finally:
        os.kill(int((tmp_path / "job.pid").read_text()), signal.SIGKILL)


def test_a_live_key_starts_hands_and_polar_hears_only_the_key_and_the_organization(opened: Open, tmp_path: Path, polar: Polar) -> None:
    kept(tmp_path, LIVE, datetime.now(UTC) - timedelta(days=3))
    app = opened("", "exit 0")
    assert app.wait(timeout=10) == 0
    text = logged(tmp_path, "hands exited 0")
    assert "license: Polar says the key ending E304DA is live" in text and LIVE not in text
    assert polar.asked == [{"key": LIVE, "organization_id": ORGANIZATION}]
    # The check that let hands start is the one the grace period runs from.
    license = json.loads((tmp_path / "license.json").read_text())
    assert datetime.now(UTC) - datetime.fromisoformat(license["validated"]) < timedelta(minutes=1)
    assert (tmp_path / "license.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("status", "body", "said"),
    [
        (404, {"error": "ResourceNotFound", "detail": "License key is no longer active."}, "License key is no longer active."),
        (404, {"error": "ResourceNotFound", "detail": "License key has expired."}, "License key has expired."),
        (404, {"error": "ResourceNotFound", "detail": "License key not found."}, "License key not found."),
        (200, {"status": "revoked", "key": LIVE}, "the key is revoked"),
    ],
)
def test_a_key_polar_refuses_does_not_start_hands_and_the_person_is_told_why(
    opened: Open, tmp_path: Path, polar: Polar, status: int, body: object, said: str
) -> None:
    polar.status, polar.body = status, body
    app = opened("", "echo hands ran")
    text = logged(tmp_path, "license: asking for a key: ")
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=5) == 0
    assert f"Polar did not accept the license key ending E304DA; if your subscription ended, renew it and start hands again. Polar says: {said}\n" in text
    assert "started hands run" not in logged(tmp_path, "asking for a key") and LIVE not in text
    # A refused key is not kept, so no later start runs it on the grace period while Polar cannot be reached.
    assert not (tmp_path / "license.json").exists()


@pytest.mark.parametrize(
    ("status", "body"),
    [(None, None), (503, {"error": "ServiceUnavailable"}), (200, "<html>Sign in to the hotel Wi-Fi</html>"), (407, "Proxy Authentication Required"), (404, {"detail": "Not Found"})],
    ids=["unreachable", "down", "captive-portal", "proxy", "no-such-route"],
)
def test_a_key_last_seen_live_within_the_grace_period_starts_hands_while_polar_cannot_be_reached(
    opened: Open, tmp_path: Path, polar: Polar, status: int | None, body: object
) -> None:
    kept(tmp_path, LIVE, datetime.now(UTC) - timedelta(days=13, hours=1))
    polar.status, polar.body = status, body
    app = opened("", "exit 0")
    assert app.wait(timeout=30) == 0
    text = logged(tmp_path, "hands exited 0")
    assert "could not reach Polar (" in text and "0 whole days of grace left" in text
    # Running on the grace period does not restart it.
    assert json.loads((tmp_path / "license.json").read_text())["validated"].startswith((datetime.now(UTC) - timedelta(days=13, hours=1)).strftime("%Y-%m-%dT%H"))


def test_a_key_not_seen_live_within_the_grace_period_does_not_start_hands_while_polar_cannot_be_reached(opened: Open, tmp_path: Path, polar: Polar) -> None:
    kept(tmp_path, LIVE, datetime.now(UTC) - timedelta(days=14, hours=1))
    polar.status, polar.body = 503, {"error": "ServiceUnavailable"}
    app = opened("", "echo hands ran")
    text = logged(tmp_path, "license: asking for a key: ")
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=5) == 0
    assert "hands has not been able to check your subscription with Polar since" in text and "HTTP 503, with no word from Polar on the key" in text
    assert "started hands run" not in logged(tmp_path, "asking for a key")


def test_with_no_key_kept_the_person_is_asked_for_one_and_polar_is_not(opened: Open, tmp_path: Path, polar: Polar) -> None:
    (tmp_path / "license.json").unlink()
    app = opened("", "echo hands ran")
    text = logged(tmp_path, "license: asking for a key: ")
    app.send_signal(signal.SIGTERM)
    assert app.wait(timeout=5) == 0
    assert "license: asking for a key: none is kept yet" in text and polar.asked == []
