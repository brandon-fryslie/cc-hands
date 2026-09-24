# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs and loads its names lazily, so a static check cannot see them; this file is the only one
# that touches CoreAudio.
"""Word from CoreAudio when the system's default input or output device changes."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import CoreAudio

_DEFAULTS = (CoreAudio.kAudioHardwarePropertyDefaultInputDevice, CoreAudio.kAudioHardwarePropertyDefaultOutputDevice)


@contextmanager
def default_device_changes(changed: Callable[[], object]) -> Iterator[None]:
    """Call `changed` whenever the default input or output device changes, for as long as the context is open.

    It is called on a CoreAudio thread, within milliseconds: measured at 17 ms from an aggregate device being
    destroyed to the call, once for the input and once for the output.
    """

    def listener(_count: int, _addresses: object) -> None:
        changed()

    addresses = [
        CoreAudio.AudioObjectPropertyAddress(selector, CoreAudio.kAudioObjectPropertyScopeGlobal, CoreAudio.kAudioObjectPropertyElementMain)
        for selector in _DEFAULTS
    ]
    for address in addresses:
        _check(CoreAudio.AudioObjectAddPropertyListenerBlock(CoreAudio.kAudioObjectSystemObject, address, None, listener), "listen for")
    try:
        yield
    finally:
        for address in addresses:
            _check(CoreAudio.AudioObjectRemovePropertyListenerBlock(CoreAudio.kAudioObjectSystemObject, address, None, listener), "stop listening for")


def _check(status: int, doing: str) -> None:
    # [LAW:no-silent-failure] a listener CoreAudio refused is a device loss nobody would hear about.
    if status != 0:
        raise OSError(f"CoreAudio would not {doing} default device changes: OSStatus {status}")
