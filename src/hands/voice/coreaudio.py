# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs and loads its names lazily, so a static check cannot see them; this file is the only one
# that touches CoreAudio.
"""The system's default audio devices, from CoreAudio, and word when they change."""

import itertools
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import CoreAudio
import objc
from loguru import logger

_SYSTEM = CoreAudio.kAudioObjectSystemObject


@dataclass(frozen=True)
class DefaultDevices:
    """CoreAudio's ids for the default input and output devices."""

    input: int
    output: int


def _address(selector: int) -> object:
    return CoreAudio.AudioObjectPropertyAddress(selector, CoreAudio.kAudioObjectPropertyScopeGlobal, CoreAudio.kAudioObjectPropertyElementMain)


_INPUT = _address(CoreAudio.kAudioHardwarePropertyDefaultInputDevice)
_OUTPUT = _address(CoreAudio.kAudioHardwarePropertyDefaultOutputDevice)


def default_devices() -> DefaultDevices:
    return DefaultDevices(_device(_INPUT), _device(_OUTPUT))


def _device(address: object) -> int:
    status, _size, data = CoreAudio.AudioObjectGetPropertyData(_SYSTEM, address, 0, b"", 4, None)
    _check(status, "say which device is the default")
    return struct.unpack("I", data)[0]


# [LAW:no-shared-mutable-globals] owned by default_device_changes alone, which adds a target when it registers its
# listener and deletes it after the listener is removed, so the listener never looks up a token that is gone.
# CoreAudio carries only an integer to the listener, so the token is how a call finds its target.
_targets: dict[int, Callable[[], object]] = {}
_tokens = itertools.count(1)


@objc.callbackFor(CoreAudio.AudioObjectAddPropertyListener)
def _listener(_object: int, _count: int, _addresses: object, token: int) -> int:
    # A module-level function and not a block: PyObjC makes a new block for every call it is passed to, so a
    # block cannot be removed again (measured: it went on being called after its removal returned 0).
    _targets[token]()
    return 0


@contextmanager
def default_device_changes(changed: Callable[[], object]) -> Iterator[None]:
    """Call `changed` whenever the default input or output device changes, for as long as the context is open.

    It is called on a CoreAudio thread, within milliseconds: measured at 17 ms from an aggregate device being
    destroyed to the call, once for the input and once for the output.
    """
    token = next(_tokens)
    _targets[token] = changed
    listening: list[object] = []
    try:
        for address in (_INPUT, _OUTPUT):
            _check(CoreAudio.AudioObjectAddPropertyListener(_SYSTEM, address, _listener, token), "listen for changes to the default device")
            listening.append(address)
        yield
    finally:
        # Only what was added is removed, so a refused registration is the error that is reported.
        refused = [CoreAudio.AudioObjectRemovePropertyListener(_SYSTEM, address, _listener, token) for address in listening]
        if any(refused):
            # [LAW:no-silent-failure] logged, not raised: raising here would turn a clean stop into a failed run. The
            # target stays, because a listener CoreAudio kept would otherwise look up a token that is gone.
            logger.error(f"CoreAudio would not stop listening for changes to the default device: OSStatus {refused}")
        else:
            del _targets[token]


def _check(status: int, doing: str) -> None:
    # [LAW:no-silent-failure] a listener CoreAudio refused is a device loss nobody would hear about.
    if status != 0:
        raise OSError(f"CoreAudio would not {doing}: OSStatus {status}")
