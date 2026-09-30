"""The summary store: a sentence for any content-addressed thing, kept until the thing changes, served by read_backlog and read_ticket."""

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from hands.core.events import Joined
from hands.core.sentences import Due, Thing, answered, digest, page, reckon
from hands.core.session import Membership, SessionId
from hands.sessions.audit import BacklogUnread, Entry, Summarised
from hands.sessions.backlog import BACKLOG, Unread, parse_export, read_backlog
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.sentences import Sentences
from hands.voice.summarising import summarise_backlog
from hands.voice.sentences import Backlog, SummaryStore
from hands.voice.tools import backlog_tools

SID = SessionId("s1")


def ticket(id: str, rank: str, title: str, description: str = "", status: str | None = "open", kind: str = "task") -> dict[str, object]:
    issue: dict[str, object] = {"id": id, "title": title, "description": description, "issue_type": kind, "rank": rank}
    return issue if status is None else {**issue, "status": status}


def child(of: str, id: str) -> dict[str, object]:
    return {"src_id": id, "dst_id": of, "type": "parent-child", "created_at": "", "created_by": ""}


# An epic with two open children and one closed, a loose ticket, a closed loose ticket, and an epic whose every child is closed.
ISSUES = [
    ticket("e1", "01", "The wire", "Put a proxy between the brain and the API.", status=None, kind="epic"),
    ticket("e1.a", "02", "The proxy", "Forward every request."),
    ticket("e1.b", "03", "The tail", "Append the sessions' state.", status="in_progress"),
    ticket("e1.c", "04", "Settings", "Which settings replace bare.", status="closed"),
    ticket("t1", "05", "Fix the flaky test", "It fails one run in ten."),
    ticket("t2", "06", "Old bug", "Already fixed.", status="closed"),
    ticket("e2", "07", "Done epic", "All of it landed.", status=None, kind="epic"),
    ticket("e2.a", "08", "Landed", "Shipped.", status="closed"),
]
RELATIONS = [child("e1", "e1.a"), child("e1", "e1.b"), child("e1", "e1.c"), child("e2", "e2.a")]
COMMENTS = [{"id": "c1", "issue_id": "t1", "body": "Seen again on CI.", "created_at": "2026-09-29T10:00:00Z", "created_by": "claude"}]


def export(issues: list[dict[str, object]] = ISSUES, relations: list[dict[str, object]] = RELATIONS) -> bytes:
    return json.dumps({"version": 2, "workspace_id": "w", "issues": issues, "relations": relations, "comments": COMMENTS, "labels": [], "events": []}).encode()


# --- keys: what changes a sentence, and what does not


def tree(a: str = "Forward every request.") -> Thing:
    return Thing("root", "", (Thing("e1", "The wire", (Thing("e1.a", a, ()), Thing("e1.b", "The tail", ()))), Thing("t1", "Flaky", ())))


def keys(thing: Thing, known: dict[str, str]) -> dict[str, str]:
    """Every thing's key once everything is said: the store filled round by round, as the task fills it."""
    by_digest: dict[str, str] = {}
    ids: dict[str, str] = {}
    while (reckoning := reckon(thing, "v1", lambda key: by_digest.get(key) or known.get(key))).due:
        for due in reckoning.due:
            by_digest[due.digest] = f"said {due.id} {due.digest[:8]}"
            ids[due.id] = due.digest
    return ids


def test_leaves_are_due_first_and_a_parent_only_once_every_part_is_said() -> None:
    first = reckon(tree(), "v1", lambda _key: None)
    assert sorted(due.id for due in first.due) == ["e1.a", "e1.b", "t1"]
    assert first.waiting == 2 and first.said == {}
    leaves = {due.digest: f"said {due.id}" for due in first.due}
    second = reckon(tree(), "v1", leaves.get)
    assert [due.id for due in second.due] == ["e1"]
    # The parent is shown each part's sentence, and keyed by them.
    assert second.due[0].parts == (("e1.a", "said e1.a"), ("e1.b", "said e1.b"))
    assert second.waiting == 1


def test_editing_one_thing_changes_its_key_and_those_above_it_and_no_other() -> None:
    before = keys(tree(), {})
    after = keys(tree(a="Forward every request, and stream the reply."), {})
    assert {id for id in before if before[id] != after[id]} == {"e1.a", "e1", "root"}


def test_an_edit_whose_new_sentence_is_the_old_one_changes_nothing_above_it() -> None:
    # Parents are keyed by their parts' sentences, not their parts' text: a sentence that did not change is no news upward.
    old = reckon(Thing("e1.a", "Forward every request.", ()), "v1", lambda _key: None).due[0]
    new = reckon(Thing("e1.a", "Forward every request!", ()), "v1", lambda _key: None).due[0]
    same = {old.digest: "Forwards requests.", new.digest: "Forwards requests."}
    assert old.digest != new.digest
    parent = [reckon(Thing("e1", "The wire", (Thing("e1.a", text, ()),)), "v1", same.get).due[0].digest for text in ("Forward every request.", "Forward every request!")]
    assert parent[0] == parent[1]


def test_a_rerank_changes_no_key() -> None:
    reranked = Thing("root", "", (Thing("t1", "Flaky", ()), Thing("e1", "The wire", (Thing("e1.b", "The tail", ()), Thing("e1.a", "Forward every request.", ())))))
    assert keys(reranked, {}) == keys(tree(), {})


def test_a_new_summariser_version_is_a_new_key_for_everything() -> None:
    assert digest("v1", "text", []) != digest("v2", "text", [])


# --- the summariser's page and its reply


def dues(*ids: str) -> list[Due]:
    return [Due(id, digest("v1", id, []), f"text of {id}", ()) for id in ids]


def test_the_page_names_each_item_and_cuts_a_long_text_in_its_middle() -> None:
    [due] = dues("t1")
    shown = page([Due("e1", due.digest, "a" * 20 + "x" * 20 + "z" * 10, (("e1.a", "A sentence."),))], text_limit=10)
    assert shown == '<item id="e1">\naaaaa\n[40 characters left out]\nzzzzz\n<parts>\n- e1.a: A sentence.\n</parts>\n</item>'


def test_a_reply_gives_each_id_its_sentence_and_says_what_it_missed_and_what_it_made_up() -> None:
    batch = dues("a", "b", "c")
    answer = answered("a: First.\n\nb: Second.\nz: Not asked.\nchatter", batch)
    assert answer.said == {batch[0].digest: "First.", batch[1].digest: "Second."}
    assert answer.missing == ("c",)
    assert answer.stray == ("z: Not asked.", "chatter")


def test_an_id_said_twice_is_said_by_neither() -> None:
    batch = dues("a")
    answer = answered("a: One.\na: Two.", batch)
    assert answer.said == {} and answer.missing == ("a",) and answer.stray == ("a: One.", "a: Two.")


# --- the backlog lit exports


def test_the_backlog_is_every_unfinished_root_in_rank_order_over_its_unfinished_children() -> None:
    backlog = parse_export(export())
    assert backlog.roots() == ("e1", "t1")
    assert backlog.open_children("e1") == ("e1.a", "e1.b")
    # An epic has no status in lit's export: it is done when every child is.
    assert backlog.done("e2") and not backlog.done("e1")
    thing = backlog.thing()
    assert thing.id == BACKLOG and [part.id for part in thing.parts] == ["e1", "t1"]
    assert [part.id for part in thing.parts[0].parts] == ["e1.a", "e1.b"]
    assert thing.parts[1].text == "Fix the flaky test\n\nIt fails one run in ten."


def test_a_follow_up_filed_under_a_closed_ticket_stands_as_a_root() -> None:
    backlog = parse_export(export(issues=[*ISSUES, ticket("t2.f", "09", "Follow up", "What the fix left.")], relations=[*RELATIONS, child("t2", "t2.f")]))
    assert backlog.roots() == ("e1", "t1", "t2.f")
    assert [part.id for part in backlog.thing().parts] == ["e1", "t1", "t2.f"]


def test_a_child_whose_parent_the_export_left_out_stands_as_a_root() -> None:
    backlog = parse_export(export(relations=[*RELATIONS, child("gone", "t1")]))
    assert "t1" in backlog.roots()


@pytest.mark.parametrize(
    "raw, why",
    [
        (json.dumps({"version": 3, "issues": [], "relations": [], "comments": []}).encode(), "version 3"),
        (export(issues=[ticket("t1", "01", "x", status="wontfix")]), "status"),
        (b"not json", "not JSON"),
    ],
)
def test_an_export_hands_does_not_read_is_refused(raw: bytes, why: str) -> None:
    with pytest.raises(Rejected, match=why):
        parse_export(raw)


# --- the pass: lit read through its own command, the summariser asked, the store filled


class Summariser:
    """A summariser that writes `said <id> <n>` for each item it is asked, `n` counting its calls, and remembers every page."""

    def __init__(self, first: int = 1) -> None:
        self.pages: list[str] = []
        self.first = first

    async def __call__(self, page: str) -> str:
        self.pages.append(page)
        return "\n".join(f"{id}: said {id} {self.first + len(self.pages) - 1}" for id in self.asked(page))

    @staticmethod
    def asked(page: str) -> list[str]:
        return re.findall(r'<item id="([^"]+)">', page)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project directory whose `lit export` hands over whatever its export.json holds."""
    bin = tmp_path / "bin"
    bin.mkdir()
    lit = bin / "lit"
    lit.write_text('#!/bin/sh\n[ "$1" = export ] || exit 2\ncat export.json\n')
    lit.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:{Path('/bin')}:{Path('/usr/bin')}")
    root = tmp_path / "project"
    root.mkdir()
    (root / "export.json").write_bytes(export())
    return root


async def test_a_pass_says_the_whole_backlog_leaves_first_and_audits_itself(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    summarise = Summariser()
    records: list[Entry] = []
    await summarise_backlog(project, store, summarise, records.append, batch=2)
    # Three leaves in two batches, then the epic, then the backlog over the epic and the loose ticket.
    assert [summarise.asked(page) for page in summarise.pages] == [["e1.a", "e1.b"], ["t1"], ["e1"], [BACKLOG]]
    assert set(store.reckon(parse_export(export()).thing()).said) == {"e1.a", "e1.b", "t1", "e1", BACKLOG}
    [record] = records
    assert isinstance(record, Summarised)
    assert (record.outcome, record.things, record.known, record.said, record.unsaid, record.rounds, record.calls, record.failed_calls, record.left_out, record.stray) == ("said", 5, 0, 5, 0, 3, 4, 0, (), 0)


async def test_editing_one_ticket_resays_only_it_and_what_sits_above_it(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    await summarise_backlog(project, store, Summariser(), lambda _entry: None)
    edited = [ticket("e1.a", "02", "The proxy", "Forward every request, and stream the reply.") if issue["id"] == "e1.a" else issue for issue in ISSUES]
    (project / "export.json").write_bytes(export(issues=edited))
    # Numbered on from the first pass, so the edited ticket's sentence is new, as a real summariser's would be.
    summarise = Summariser(first=100)
    records: list[Entry] = []
    await summarise_backlog(project, store, summarise, records.append)
    assert [summarise.asked(page) for page in summarise.pages] == [["e1.a"], ["e1"], [BACKLOG]]
    [record] = records
    assert isinstance(record, Summarised) and (record.known, record.said) == (2, 3)


async def test_a_rerank_or_a_status_change_asks_the_summariser_nothing(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    await summarise_backlog(project, store, Summariser(), lambda _entry: None)
    moved = [{**issue, "rank": "00"} if issue["id"] == "t1" else {**issue, "status": "open"} if issue["id"] == "e1.b" else issue for issue in ISSUES]
    (project / "export.json").write_bytes(export(issues=moved))
    summarise = Summariser()
    await summarise_backlog(project, store, summarise, lambda _entry: None)
    assert summarise.pages == []


async def test_what_the_summariser_leaves_out_stays_unsaid_and_the_pass_says_so(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))

    async def forgets_t1(page: str) -> str:
        return "\n".join(f"{id}: said {id}" for id in Summariser.asked(page) if id != "t1")

    records: list[Entry] = []
    await summarise_backlog(project, store, forgets_t1, records.append)
    # The epic is said; the backlog, keyed by t1's sentence too, cannot be.
    assert set(store.reckon(parse_export(export()).thing()).said) == {"e1.a", "e1.b", "e1"}
    [record] = records
    assert isinstance(record, Summarised) and (record.outcome, record.said, record.unsaid) == ("partial", 3, 2)
    # Asked in every round, left out of every reply: the line tells a model that skips an item from calls that failed.
    assert (record.rounds, record.left_out, record.failed_calls) == (3, ("t1", "t1", "t1"), 0)


async def test_a_project_lit_cannot_read_is_audited_as_unread_and_why(project: Path, tmp_path: Path) -> None:
    (project / "export.json").unlink()
    records: list[Entry] = []
    await summarise_backlog(project, SummaryStore(Sentences(tmp_path / "sentences.db")), Summariser(), records.append)
    [record] = records
    assert isinstance(record, BacklogUnread) and "exited 1" in record.error


async def test_an_export_that_hangs_is_unread_and_its_process_is_not_left_running(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lit = Path(os.environ["PATH"].split(":")[0]) / "lit"
    lit.write_text("#!/bin/sh\necho $$ > pid\nexec sleep 30\n")
    monkeypatch.setattr("hands.sessions.backlog.EXPORT_TIMEOUT_SECONDS", 0.5)
    with pytest.raises(Unread, match="did not answer"):
        await read_backlog(project)
    with pytest.raises(ProcessLookupError):
        os.kill(int((project / "pid").read_text()), 0)


async def test_a_summariser_that_cannot_start_is_a_failed_call_and_the_pass_still_ends(project: Path, tmp_path: Path) -> None:
    async def no_claude(_page: str) -> str:
        raise FileNotFoundError("claude")

    records: list[Entry] = []
    await summarise_backlog(project, SummaryStore(Sentences(tmp_path / "sentences.db")), no_claude, records.append)
    [record] = records
    assert isinstance(record, Summarised) and (record.outcome, record.said, record.calls, record.failed_calls) == ("partial", 0, 1, 1)


async def test_a_backlog_with_nothing_left_asks_the_summariser_nothing(project: Path, tmp_path: Path) -> None:
    (project / "export.json").write_bytes(export(issues=[ticket("t2", "01", "Old bug", status="closed")], relations=[]))
    summarise = Summariser()
    records: list[Entry] = []
    await summarise_backlog(project, SummaryStore(Sentences(tmp_path / "sentences.db")), summarise, records.append)
    [record] = records
    assert summarise.pages == []
    assert isinstance(record, Summarised) and (record.outcome, record.things, record.calls) == ("said", 0, 0)


# --- the tools


async def tools(project: Path, store: SummaryStore) -> dict[str, Any]:
    sessions = Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None)
    await sessions.apply(Joined(Membership(SID, pid=4242, cwd=project, transcript=project / "s1.jsonl"), "startup"))
    return {tool.name: tool.body for tool in backlog_tools(sessions, store)}


async def test_read_backlog_serves_titles_until_the_sentences_are_made_and_asks_for_them(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    read_backlog = (await tools(project, store))["read_backlog"]
    before = dict(await read_backlog(session="s1"))
    assert "summary" not in before and before["unsummarised"] == 2
    assert before["items"] == [
        {"id": "e1", "title": "The wire", "children_open": 2, "children_done": 1},
        {"id": "t1", "title": "Fix the flaky test", "status": "open"},
    ]
    # The read is a sighting: the backlog is asked for, to be said off the voice path.
    assert await store.wanted() == Backlog(project)
    await summarise_backlog(project, store, Summariser(), lambda _entry: None)
    after = dict(await read_backlog(session="s1"))
    assert after["summary"] == f"said {BACKLOG} 3" and after["unsummarised"] == 0
    assert [item["summary"] for item in after["items"]] == ["said e1 2", "said t1 1"]


async def test_read_ticket_gives_the_sentence_and_what_is_open_under_it_and_its_own_words_only_when_asked(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    await summarise_backlog(project, store, Summariser(), lambda _entry: None)
    read_ticket = (await tools(project, store))["read_ticket"]
    epic = dict(await read_ticket(session="s1", ticket="e1"))
    assert epic["summary"] == "said e1 2" and "description" not in epic
    assert [item["id"] for item in epic["open_children"]] == ["e1.a", "e1.b"]
    assert epic["open_children"][1] == {"id": "e1.b", "title": "The tail", "summary": "said e1.b 1", "status": "in_progress"}
    full = dict(await read_ticket(session="s1", ticket="t1", full=True))
    assert full["description"] == "It fails one run in ten."
    assert full["comments"] == [{"by": "claude", "at": "2026-09-29T10:00:00Z", "body": "Seen again on CI."}]
    leaf = dict(await read_ticket(session="s1", ticket="e1.a"))
    assert leaf["parent"]["id"] == "e1" and leaf["open_children"] == []


async def test_the_backlog_tools_say_what_they_could_not_read(project: Path, tmp_path: Path) -> None:
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    bodies = await tools(project, store)
    assert await bodies["read_ticket"](session="s1", ticket="nope") == {"error": "the backlog has no ticket nope"}
    assert await bodies["read_backlog"](session="s9") == {"error": "there is no session s9"}
    (project / "export.json").unlink()
    assert "could not be read" in str((await bodies["read_backlog"](session="s1"))["error"])


def test_the_store_keeps_what_it_was_told_across_opening_it_again(tmp_path: Path) -> None:
    key = digest("v1", "text", [])
    Sentences(tmp_path / "sentences.db").keep({key: "Kept."})
    reopened = Sentences(tmp_path / "sentences.db")
    assert reopened.known(key) == "Kept."
    # A key already said keeps its first sentence.
    reopened.keep({key: "Rewritten."})
    assert reopened.known(key) == "Kept."
