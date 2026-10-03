"""Why a call to the language model failed, as a fact a listener can act on: read off an API variant's exception or off
the brain's wire, and said by the system channel."""

from dataclasses import dataclass

from pipecat.utils.errors import ErrorCategory

from hands.core.wire import UsageLimitReached


@dataclass(frozen=True)
class ModelUnreachable:
    pass


@dataclass(frozen=True)
class ModelFailed:
    category: ErrorCategory


@dataclass(frozen=True)
class ModelReplyEmpty:
    """The model answered with nothing: no words, and no call that would say or do something for it."""


ModelFact = ModelUnreachable | UsageLimitReached | ModelFailed | ModelReplyEmpty


class ModelFault(Exception):
    """A failure of the model already read as the fact it is, as the brain's stage reads it off the wire: carried on the
    stage's error frame so the channel says that fact."""

    def __init__(self, fact: ModelFact) -> None:
        super().__init__(fact)
        self.fact = fact
