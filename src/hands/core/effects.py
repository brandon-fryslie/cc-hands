"""What the reducer asks the edges to do. Adapters perform these and nothing else."""

from dataclasses import dataclass

from hands.core.events import SessionEvent


@dataclass(frozen=True)
class Unregistered:
    """An event named a session that never joined, so it changed nothing."""

    event: SessionEvent


AuditRecord = Unregistered


@dataclass(frozen=True)
class Audit:
    record: AuditRecord


Effect = Audit
